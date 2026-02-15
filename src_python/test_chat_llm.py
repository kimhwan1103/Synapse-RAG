import torch
import torch.nn as nn
import os
import requests
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from models import HybridQueryGNN

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
    ADAPTER_PATH = os.path.join(WEIGHT_PATH, "llm_adapter") 
    SERVER_URL = "http://rag-db-server:9000"
    
    MAX_HISTORY = 6 
    USE_LORA = False 

cfg = Config()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [유틸] 화면 출력 및 저장용 스트리머
class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.captured_text = ""
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.captured_text += text
        super().on_finalized_text(text, stream_end)

# ==============================================================================
# 🤖 추론 모델 정의
# ==============================================================================
class GraphLLM_Inference(nn.Module):
    def __init__(self):
        super().__init__()
        self.chat_history = [] 

        # 1. GNN
        self.gnn = HybridQueryGNN(cfg.EMBEDDING_DIM, cfg.GNN_HIDDEN).to(DEVICE)
        
        # 2. LLM
        print(f"🤖 Loading LLM ({cfg.LLM_MODEL_ID})...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.LLM_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_ID, trust_remote_code=True)
        
        # 3. LoRA
        if cfg.USE_LORA and os.path.exists(cfg.ADAPTER_PATH):
            print(f"🧩 LoRA Adapter 로드 중...")
            try:
                self.llm = PeftModel.from_pretrained(self.llm, cfg.ADAPTER_PATH)
                print("✅ LoRA 적용 완료.")
            except Exception as e:
                print(f"⚠️ LoRA 로드 실패: {e}")
        else:
            print("⚡ LoRA 없이 '순수 LLM' + 'Projector' 모드로 동작합니다.")
        
        # 4. Projector
        llm_hidden = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(cfg.GNN_HIDDEN, llm_hidden),
            nn.LayerNorm(llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden)
        ).to(DEVICE)
        
        self.load_weights()
        self.eval() 
        
        # ★ 전체 노드 캐시 (검색 속도 향상용)
        self.all_nodes_cache = []
        self.all_embeddings_cache = None

    def load_weights(self):
        if os.path.exists(cfg.GARDENER_PATH):
            state = torch.load(cfg.GARDENER_PATH, map_location=DEVICE)
            my_state = self.gnn.state_dict()
            for k, v in state.items():
                target_k = f"local_gcn.{k.replace('encoder.', '')}"
                if target_k in my_state:
                    my_state[target_k].copy_(v)
            print("✅ Gardener Weights Loaded.")
            
        if os.path.exists(cfg.PROJECTOR_PATH):
            self.projector.load_state_dict(torch.load(cfg.PROJECTOR_PATH, map_location=DEVICE))
            print("✅ Projector Weights Loaded.")

    # ★ [핵심 수정] 진짜 검색 + GNN 확장 로직
    def retrieve_subgraph(self, embedder, query_text):
        try:
            # 1. 쿼리 임베딩 (Client에서 수행)
            query_vec = embedder.encode(query_text).tolist() # 리스트로 변환
            query_tensor = torch.tensor(query_vec).float().to(DEVICE).unsqueeze(0)

            print("📡 서버에 검색 요청 중...", end=" ", flush=True)
            
            # 2. 서버에 벡터 보내서 검색 (Lightweight)
            res = requests.post(
                f"{cfg.SERVER_URL}/search_vector", 
                json={"vector": query_vec, "k": 1},
                timeout=5
            )
            
            if not res.ok: 
                print("서버 오류")
                return None
            
            data = res.json()
            results = data.get("results", [])
            
            if not results:
                print("검색 결과 없음")
                return None
                
            best_node = results[0]
            score = best_node['score']

            if score < 0.8:
                print(f"점수 미달 ({score:.4f} < 0.8). 검색 결과 버림")
                return None

            print(f"\n🔎 [서버 검색] '{best_node['text'][:20]}...' (Score: {best_node['score']:.4f})")

            # 3. GNN 확장 (이웃 가져오기)
            # (중심 노드의 ID를 알았으니 이웃을 조회)
            try:
                res_neighbors = requests.get(
                    f"{cfg.SERVER_URL}/get_neighbors", 
                    params={"node_id": best_node['id']},
                    timeout=5 
                )
                neighbors = res_neighbors.json() if res_neighbors.ok else []
            except: neighbors = []

            # 4. 서브그래프 구성
            # 중심 노드 (서버가 임베딩까지 줘서 그대로 쓰면 됨)
            relevant_nodes = [best_node]
            
            # 이웃 노드들의 상세 정보 가져오기 (각각 조회 or Bulk 조회)
            # 여기서는 편의상 이웃 ID로 각각 조회 (개수가 적으므로 빠름)
            for nid, weight in neighbors:
                try:
                    r = requests.get(f"{cfg.SERVER_URL}/get_node", params={"node_id": nid}, timeout=2)
                    if r.ok:
                        n_data = r.json()
                        relevant_nodes.append(n_data)
                except: pass
            
            print(f"🕸️ [GNN] 중심 노드 + 친구 {len(relevant_nodes)-1}명 연결")

            # 5. 텐서 변환 및 엣지 생성
            # (주의: 서버에서 받은 embedding이 리스트 형태이므로 텐서로 변환)
            x_list = [n['embedding'] for n in relevant_nodes]
            x = torch.tensor(x_list, dtype=torch.float).to(DEVICE)

            src, dst = [], []
            for i in range(1, len(relevant_nodes)):
                src.append(0); dst.append(i)
                src.append(i); dst.append(0)
            
            if not src: src, dst = [0], [0]
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)
            
            context_texts = [f"- {n['text']}" for n in relevant_nodes]
            
            return x, edge_index, query_tensor, "\n".join(context_texts)

        except Exception as e:
            print(f"❌ 검색 에러: {e}")
            return None
            
    def clear_history(self):
        self.chat_history = []
        print("🧹 기억 초기화 완료.")

    @torch.no_grad()
    def chat(self, query, embedder):
        if query == "!reset": self.clear_history(); return

        # 1. 검색 및 추론
        context_str = ""
        graph_embeds = None
        
        if len(query) < 2: retrieved = None
        else:
            print("🔍 GNN 검색 시작...", end=" ", flush=True)
            retrieved = self.retrieve_subgraph(embedder, query)
        
        target_device = self.llm.get_input_embeddings().weight.device
        
        if retrieved:
            x, edge_index, query_emb, context_texts = retrieved
            
            # GNN Forward
            _, h_global, _ = self.gnn(x, edge_index, query_emb)
            graph_embeds = self.projector(h_global).unsqueeze(1).to(self.llm.dtype).to(target_device) * 0.7 # 영향력 강화
            
            context_str = f"[검색된 지식]\n{context_texts}\n"
        else:
            print("⚠️ 관련 정보를 못 찾았습니다.")

        # 2. 프롬프트 구성
        system_prompt = (
            "당신은 그래프 신경망(GNN)과 거대언어모델(LLM)이 결합된 고지능 AI 연구 비서입니다. "
            "주어진 [검색된 지식]이 질문과 밀접하게 관련이 있다면 이를 근거로 답변하고, "
            "지식이 주어지지 않았거나 질문과 관련이 없다면 당신의 배경지식을 활용해 자연스럽게 대화하세요. "
            "특히 인사말이나 신변잡기적인 질문에는 친절하게 응대하세요."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.chat_history[-cfg.MAX_HISTORY:]:
            messages.append(turn)
            
        user_content = f"{context_str}\n질문: {query}"
        messages.append({"role": "user", "content": user_content})
        
        # 입력 생성
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(target_device)
        input_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
        
        # 그래프 임베딩 주입 (프롬프트 맨 앞에)
        if graph_embeds is not None:
            final_embeds = torch.cat([graph_embeds, input_embeds], dim=1)
            mask = torch.cat([torch.ones((1, 1), device=target_device), inputs.attention_mask], dim=1)
        else:
            final_embeds = input_embeds
            mask = inputs.attention_mask

        # 3. 생성
        print("\nBot: ", end="", flush=True)
        streamer = CaptureStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        self.llm.generate(
            inputs_embeds=final_embeds,
            attention_mask=mask,
            max_new_tokens=200,
            streamer=streamer,
            do_sample=True,
            temperature=0.3, # 사실 기반 답변을 위해 낮춤
            pad_token_id=self.tokenizer.eos_token_id
        )
        print() 
        
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": streamer.captured_text})

if __name__ == "__main__":
    print("📥 Embedder 로딩...")
    embedder = SentenceTransformer("jhgan/ko-sroberta-multitask", device=DEVICE)
    bot = GraphLLM_Inference()
    print("\n💬 GNN 챗봇 시작! (종료: exit)")
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]: break
        if not q.strip(): continue
        bot.chat(q, embedder)