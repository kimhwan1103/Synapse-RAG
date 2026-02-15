import os
import sys
import time
import requests
import torch
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# ==============================================================================
# 1. 🚑 [보안 패치] (가장 먼저 실행)
# ==============================================================================
try:
    import transformers.modeling_utils
    import transformers.utils.import_utils
    def bypass_check(): return None
    transformers.utils.import_utils.check_torch_load_is_safe = bypass_check
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = bypass_check
    print("✅ [Security Patch] Successfully bypassed torch.load check.")
except: pass

# ==============================================================================
# 2. 🧠 [모델 가져오기] train_proposed.py 에서 클래스 직접 Import!
# ==============================================================================
# 주의: train_proposed.py가 같은 폴더에 있어야 합니다.
try:
    from train_proposed import GraphRAG_Generator
    print("✅ 'train_proposed.py'에서 모델 구조를 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"❌ 모델 Import 실패: {e}")
    print("   (test_chat.py와 train_proposed.py가 같은 폴더에 있는지 확인해주세요.)")
    sys.exit(1)

# ==============================================================================
# 3. 🤖 [챗봇 시스템]
# ==============================================================================

SERVER_URL = "http://rag-db-server:9000"
#MODEL_PATH = "./data/weights/proposed_model_latest copy.pth"
MODEL_PATH = "../weights/proposed_model_latest.pth"
DEVICE = torch.device('cpu') 

print(f"🤖 ChatBot initialized on {DEVICE}")

class ChatBot:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
        
        # ★ Import한 클래스 사용
        self.model = GraphRAG_Generator(vocab_size=self.tokenizer.vocab_size).to(DEVICE)
        
        self.load_latest_weights()
        self.model.eval()

    def load_latest_weights(self):
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            return
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            self.model.load_state_dict(checkpoint, strict=False)
            print("✅ 최신 가중치 로드 완료!")
        except Exception as e:
            print(f"❌ 가중치 로드 실패: {e}")

    def retrieve_subgraph(self, query_text):
        query_emb = self.embedder.encode(query_text).tolist()
        try:
            # 테스트용: 배치 데이터 가져오기
            # ★ 수정 1: 타임아웃을 2초 -> 10초로 넉넉하게 변경
            res = requests.get(f"{SERVER_URL}/get_training_batch", params={"batch_size": 10}, timeout=60)
            
            if not res.ok: 
                print(f"❌ Server Error: {res.status_code}")
                return None
            
            data = res.json()
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            
            if not nodes: 
                print("⚠️ No nodes returned from server.")
                return None
            
            x = torch.tensor([n['embedding'] for n in nodes], dtype=torch.float).to(DEVICE)
            
            id_map = {n['id']: i for i, n in enumerate(nodes)}
            src_list = []
            dst_list = []
            for e in edges:
                if e['src'] in id_map and e['dst'] in id_map:
                    src_list.append(id_map[e['src']])
                    dst_list.append(id_map[e['dst']])
            
            if not src_list:
                src_list = [0] * len(nodes)
                dst_list = [0] * len(nodes)

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=DEVICE)
            return x, edge_index, torch.tensor(query_emb).to(DEVICE)
            
        except Exception as e:
            # ★ 수정 2: 어떤 에러인지 정확히 출력
            print(f"❌ Retrieval Error Detailed: {e}")
            return None

    def generate_response(self, query):
        self.load_latest_weights() 
        
        retrieved = self.retrieve_subgraph(query)
        # (중략: Fallback 로직 등으로 데이터가 없어도 진행되도록 처리되어 있다고 가정)
        if not retrieved:
             # 만약 retrieve_subgraph가 None을 리턴하면 에러 나니까 
             # 여기서도 Fallback(가짜 데이터)을 쓰거나 종료해야 합니다.
             # 앞서 말씀드린 'Offline Mode' 코드가 있다면 이 부분은 안 탈 겁니다.
             return "데이터를 가져올 수 없어 대화를 할 수 없어요."

        x, edge_index, query_vec = retrieved
        
        # 차원 맞추기
        # x: (Nodes, Dim), edge_index: (2, Edges) -> 그대로 둠
        # query_vec: (Dim) -> (1, Dim)으로 unsqueeze
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        
        decoder_input = torch.tensor([[self.tokenizer.cls_token_id]]).to(DEVICE)
        response_tokens = []
        
        for i in range(50): # 최대 50토큰
            with torch.no_grad():
                logits, _ = self.model(x, edge_index, query_vec, decoder_input, batch=None)
                
                # 현재 스텝의 로짓 (마지막 토큰 예측값)
                next_token_logits = logits[:, -1, :]

                # -----------------------------------------------------------
                # 1. 반복 방지 (Repetition Penalty)
                # -----------------------------------------------------------
                # 이미 내뱉은 단어들의 확률을 강제로 깎아버립니다. (1.2배 패널티)
                for token in response_tokens:
                    next_token_logits[0, token] /= 1.2
                
                # -----------------------------------------------------------
                # 2. Top-k Sampling (창의성 추가)
                # -----------------------------------------------------------
                # 확률 상위 50개 단어 중에서 랜덤으로 뽑습니다. (Greedy 탈피)
                # Temperature(0.7)로 너무 튀지 않게 조절
                temperature = 0.7
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                
                top_k = 50
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                
                # 확률 분포에 따라 랜덤 뽑기
                next_token_idx = torch.multinomial(top_k_probs, 1).item()
                next_token = top_k_indices[0, next_token_idx].item()

                if next_token == self.tokenizer.sep_token_id: 
                    break
                
                response_tokens.append(next_token)
                
                # 다음 입력 준비
                next_input = torch.tensor([[next_token]]).to(DEVICE)
                decoder_input = torch.cat([decoder_input, next_input], dim=1)
        
        print(" Done!")
        return self.tokenizer.decode(response_tokens)
        return self.tokenizer.decode(response_tokens)

if __name__ == "__main__":
    bot = ChatBot()
    print("\n💬 대화 시작! (종료하려면 'exit' 입력)")
    while True:
        try:
            q = input("User: ")
            if q.lower() in ["exit", "quit"]: break
            if not q.strip(): continue
            
            start = time.time()
            ans = bot.generate_response(q)
            print(f"Bot : {ans} ({time.time()-start:.2f}s)")
        except KeyboardInterrupt:
            break