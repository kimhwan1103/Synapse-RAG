import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import time
import requests
import random
import math
from models import HybridQueryGNN
from torch_geometric.utils import to_dense_batch
from transformers import BertTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Trainer] %(message)s')
logger = logging.getLogger("Trainer")

WEIGHT_PATH = "../weights"
GARDENER_WEIGHT_PATH = os.path.join(WEIGHT_PATH, "gardener_encoder_latest.pth")
MY_MODEL_PATH = os.path.join(WEIGHT_PATH, "proposed_model_latest.pth")
SERVER_URL = "http://rag-db-server:9000"

# ==============================================================================
# 1. Full Model 정의
# ==============================================================================
class GraphRAG_Generator(nn.Module):
    def __init__(self, node_dim=768, hidden_dim=256, vocab_size=30522, num_heads=4):
        # BERT Vocab Size는 보통 30522
        super().__init__()
        self.encoder = HybridQueryGNN(node_dim, hidden_dim, num_heads)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4,
            batch_first=True,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, edge_index, query_emb, target_ids, batch=None):
        h_nodes, h_global, attn_weights = self.encoder(x, edge_index, query_emb, batch)
        
        from torch_geometric.utils import to_dense_batch
        h_nodes_dense, mask = to_dense_batch(h_nodes, batch)
        
        h_global = h_global.unsqueeze(1)
        memory = torch.cat([h_global, h_nodes_dense], dim=1)
        
        global_mask = torch.ones((mask.size(0), 1), device=mask.device).bool()
        full_mask = torch.cat([global_mask, mask], dim=1)
        
        tgt = self.embedding(target_ids)
        tgt = self.pos_encoder(tgt)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~full_mask
        )
        
        return self.fc_out(output), attn_weights

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def load_gardener_weights(self):
        if not os.path.exists(GARDENER_WEIGHT_PATH):
            logger.warning("⚠️ No Gardener weights found yet.")
            return

        logger.info(f"🔄 Loading Gardener weights...")
        try:
            gardener_state = torch.load(GARDENER_WEIGHT_PATH)
            my_state = self.state_dict()
            # 간단한 키 매칭 로직 (이름에 conv가 포함되면 복사)
            for k, v in gardener_state.items():
                for my_k in my_state.keys():
                    if k in my_k and v.shape == my_state[my_k].shape:
                        my_state[my_k].copy_(v)
            logger.info("✅ Weight Transfer Complete (Partial).")
        except Exception as e:
            logger.error(f"Transfer failed: {e}")

import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

# ==============================================================================
# 2. 데이터 페칭 (Real Data from Server)
# ==============================================================================
def fetch_real_batch(device, tokenizer):
    try:
        # 배치 사이즈 약간 늘림
        res = requests.get(f"{SERVER_URL}/get_training_batch", params={"batch_size": 4096})
        if not res.ok: return None
        
        data = res.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # 1. 텍스트가 있는 유효한 노드만 필터링 (text 필드 확인)
        valid_nodes = [n for n in nodes if len(n["embedding"]) == 768 and "text" in n and len(n["text"]) > 5]
        
        if len(valid_nodes) < 2: return None

        # 2. 타겟(정답) 노드 하나 랜덤 선택
        target_node = random.choice(valid_nodes)
        
        # ★ [수정] 진짜 텍스트 사용!
        target_text = target_node["text"] 
        
        # ID 매핑 및 텐서 변환 (동일)
        id_to_idx = {n["id"]: i for i, n in enumerate(valid_nodes)}
        x = torch.tensor([n["embedding"] for n in valid_nodes], dtype=torch.float).to(device)
        
        # ... (엣지 구성 로직 동일) ...
        src_list, dst_list = [], []
        for e in edges:
            if e["src"] in id_to_idx and e["dst"] in id_to_idx:
                src_list.append(id_to_idx[e["src"]])
                dst_list.append(id_to_idx[e["dst"]])
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        if edge_index.size(1) == 0:
             edge_index = torch.stack([torch.arange(len(valid_nodes)), torch.arange(len(valid_nodes))]).to(device)

        # 3. 질문(Query) 생성
        target_emb = torch.tensor(target_node["embedding"], dtype=torch.float).to(device)
        noise = torch.randn_like(target_emb) * 0.1
        query_emb = (target_emb + noise).unsqueeze(0)

        # 4. 정답 토크나이징 (진짜 텍스트)
        # max_length를 좀 늘려서(예: 64) 더 긴 문장 학습
        tokens = tokenizer(target_text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
        target_ids = tokens.input_ids.to(device)
        
        batch = torch.zeros(len(valid_nodes), dtype=torch.long).to(device)
        
        return x, edge_index, query_emb, target_ids, batch
    
    except Exception as e:
        return None

# ==============================================================================
# 3. 학습 루프
# ==============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ... (초기화 코드 동일) ...
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = GraphRAG_Generator(vocab_size=tokenizer.vocab_size).to(device)
    
    # 처음에 한 번 로드
    model.load_gardener_weights()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 학습률 약간 낮춤 (안정성)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    model.train()
    
    for step in range(1, 100001): # 스텝 수 늘림
        data = fetch_real_batch(device, tokenizer)
        if not data:
            time.sleep(0.5)
            continue
            
        x, edge_index, query_emb, target_ids, batch = data
            
        decoder_input = target_ids[:, :-1]
        target_output = target_ids[:, 1:]
        
        # Forward
        logits, attn_weights = model(x, edge_index, query_emb, decoder_input, batch)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # --- 로그 출력 (기존 동일) ---
        if step % 10 == 0:
            with torch.no_grad():
                ppl = torch.exp(loss)
                preds = torch.argmax(logits, dim=-1)
                mask = target_output != tokenizer.pad_token_id
                acc = (preds[mask] == target_output[mask]).float().mean() * 100
                gnn_focus = attn_weights.max().item()
                logger.info(f"[Step {step}] Loss: {loss.item():.4f} | PPL: {ppl.item():.2f} | Acc: {acc.item():.2f}% | Focus: {gnn_focus:.4f}")

        # --- 모델 저장 및 서버 청소 (기존 동일) ---
        if step % 500 == 0:
            torch.save(model.state_dict(), MY_MODEL_PATH)
            logger.info(f"💾 Model checkpoint saved.")
            try: requests.post(f"{SERVER_URL}/cleanup_memory", timeout=1)
            except: pass

        # ======================================================================
        # ★ [NEW] Gardener와 뇌 동기화 (Brain Sync)
        # ======================================================================
        # Gardener가 100스텝마다 저장하므로, 우리는 200~500스텝마다 가져오면 적당함
        if step % 200 == 0:
            logger.info(f"🔄 Syncing with Gardener's latest brain...")
            
            # 현재 학습 중인 옵티마이저 상태 보존을 위해 잠시 no_grad
            try:
                model.load_gardener_weights()
                # 주의: 가중치가 갑자기 바뀌면 Loss가 잠깐 튈 수 있지만, 
                # 장기적으로는 더 똑똑한 GNN Encoder를 쓰게 되므로 이득입니다.
            except Exception as e:
                logger.warning(f"Sync failed (skip): {e}")

if __name__ == "__main__":
    train()