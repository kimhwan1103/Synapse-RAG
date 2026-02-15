import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import HybridQueryGNN
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ==============================================================================
# ⚙️ 설정
# ==============================================================================
class Config:
    LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    EMBEDDING_DIM = 768
    GNN_HIDDEN = 768
    
    WEIGHT_PATH = "/workspace/weights"
    GARDENER_PATH = os.path.join(WEIGHT_PATH, "gardener_encoder_latest.pth")
    PROJECTOR_PATH = os.path.join(WEIGHT_PATH, "graph_projector_latest.pth")
    SNAPSHOT_FILE = "train_snapshot_clean.pt" 
    
    BATCH_SIZE = 4
    ACCUM_STEPS = 4
    LEARNING_RATE = 2e-4
    MAX_STEPS = 2000
    SAVE_INTERVAL = 100
    
    USE_LORA = True
    LORA_RANK = 8
    LORA_ALPHA = 32

cfg = Config()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Trainer")

# ==============================================================================
# 💿 데이터셋
# ==============================================================================
class GraphDataset(Dataset):
    def __init__(self, snapshot_path):
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"❌ 데이터 파일이 없습니다: {snapshot_path}")
        logger.info(f"📂 데이터 파일 로딩 중... ({snapshot_path})")
        data = torch.load(snapshot_path)
        self.nodes = data['nodes']
        logger.info(f"✅ 로딩 완료! 총 {len(self.nodes)}개 샘플")

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes[idx]
        return {
            "embedding": torch.tensor(node['embedding'], dtype=torch.float),
            "text": node['text']
        }

# ★ [수정됨] 배치 생성 함수 (에러 수정 완료)
def graph_collate_fn(batch_list):
    batch_size = len(batch_list)
    
    # 1. 노드 임베딩
    x = torch.stack([item['embedding'] for item in batch_list]) # [Batch, 768]
    
    # 2. 엣지 생성 (각 샘플 독립적 처리)
    src = list(range(batch_size))
    dst = list(range(batch_size))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 3. 텍스트
    text_list = [item['text'] for item in batch_list]
    
    # 4. Query Embedding
    # ★ [수정] unsqueeze(1) 제거! GRU는 2D 입력(Batch, Hidden)을 원함
    query_emb = x.clone() # [Batch, 768]
    query_emb = query_emb + torch.randn_like(query_emb) * 0.1 # 노이즈 추가
    
    # 5. Batch Index
    batch_idx = torch.arange(batch_size, dtype=torch.long)
    
    return x, edge_index, query_emb, text_list, batch_idx

# ==============================================================================
# 🏗️ 모델 정의
# ==============================================================================
class GraphLLM_Connector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = HybridQueryGNN(cfg.EMBEDDING_DIM, cfg.GNN_HIDDEN)
        
        logger.info(f"🤖 Loading LLM ({cfg.LLM_MODEL_ID})...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.LLM_MODEL_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_ID, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if cfg.USE_LORA:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]
            )
            self.llm = get_peft_model(self.llm, peft_config)
            
        llm_hidden = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(cfg.GNN_HIDDEN, llm_hidden), nn.LayerNorm(llm_hidden), nn.GELU(), nn.Linear(llm_hidden, llm_hidden)
        )
        self._freeze_parameters()

    def _freeze_parameters(self):
        for param in self.gnn.parameters(): param.requires_grad = False
        for param in self.projector.parameters(): param.requires_grad = True
        logger.info("❄️ GNN Frozen. Projector & LoRA are trainable.")

    def load_gardener_weights(self):
        if os.path.exists(cfg.GARDENER_PATH):
            try:
                state = torch.load(cfg.GARDENER_PATH, map_location="cpu", weights_only=True)
            except:
                state = torch.load(cfg.GARDENER_PATH, map_location="cpu")
            
            my_state = self.gnn.state_dict()
            for k, v in state.items():
                target_k = f"local_gcn.{k.replace('encoder.', '')}"
                if target_k in my_state: my_state[target_k].copy_(v)

    def forward(self, x, edge_index, query_emb, target_text_list, batch=None):
        # 1. GNN Forward
        _, h_global, _ = self.gnn(x, edge_index, query_emb, batch)
        
        # 2. Projector (GNN Vector -> LLM Vector)
        # h_global: [Batch, Hidden]
        graph_embeds = self.projector(h_global).unsqueeze(1).to(self.llm.dtype) # [Batch, 1, LLM_Hidden]
        
        # 3. Text Prompting
        prompts = [f"<|im_start|>user\nContext를 보고 설명해.\n<|im_end|>\n<|im_start|>assistant\n{t}<|im_end|>" for t in target_text_list]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.llm.device)
        
        # 4. Concatenate
        input_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
        combined_embeds = torch.cat([graph_embeds, input_embeds], dim=1) 
        
        # 5. Masking & Labels
        ones = torch.ones((combined_embeds.shape[0], 1), device=self.llm.device)
        combined_mask = torch.cat([ones, inputs.attention_mask], dim=1)
        
        ignore = torch.full((combined_embeds.shape[0], 1), -100, device=self.llm.device)
        labels = torch.cat([ignore, inputs.input_ids], dim=1)
        
        return self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels).loss

# ==============================================================================
# 🔥 학습 루프
# ==============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.WEIGHT_PATH, exist_ok=True)
    
    try:
        dataset = GraphDataset(cfg.SNAPSHOT_FILE)
    except Exception as e:
        logger.error(str(e))
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=graph_collate_fn,
        num_workers=4,
        pin_memory=True 
    )
    
    model = GraphLLM_Connector().to(device)
    model.load_gardener_weights()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    logger.info("🚀 Training Loop Started! (Local Dataset)")
    model.train()
    
    step = 0
    epochs = 3 
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        accum_loss = 0
        
        for batch_data in pbar:
            step += 1
            x, edge_index, query_emb, text_list, batch_idx = batch_data
            
            x = x.to(device)
            edge_index = edge_index.to(device)
            query_emb = query_emb.to(device)
            batch_idx = batch_idx.to(device)
            
            with torch.amp.autocast('cuda'):
                loss = model(x, edge_index, query_emb, text_list, batch_idx)
            
            accum_loss += loss.item()
            scaler.scale(loss / cfg.ACCUM_STEPS).backward()
            
            if step % cfg.ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{accum_loss/cfg.ACCUM_STEPS:.4f}")
                accum_loss = 0
            
            if step % cfg.SAVE_INTERVAL == 0:
                torch.save(model.projector.state_dict(), cfg.PROJECTOR_PATH)
                model.llm.save_pretrained(os.path.join(cfg.WEIGHT_PATH, "llm_adapter"))
                
    logger.info("🎉 학습 완료!")

if __name__ == "__main__":
    train()