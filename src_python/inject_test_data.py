import requests
import json
import time
from sentence_transformers import SentenceTransformer

# 설정
SERVER_URL = "http://rag-db-server:9000"
MODEL_ID = "jhgan/ko-sroberta-multitask" 

def inject_scenario():
    print("🧪 [GNN Test] 멀티 홉 데이터 주입을 시작합니다...")
    
    # 0. 임베딩 모델 로딩
    print("   0️⃣ 임베딩 모델 로딩 중... (잠시 대기)")
    embedder = SentenceTransformer(MODEL_ID)

    # 1. 노드 데이터 정의
    text_a = "비밀 프로젝트 '블랙 펄'의 데이터는 오직 '제3 서버'에만 저장되어 있다."
    text_b = "'제3 서버'의 접속 암호는 '7777'이며, 관리자는 김철수다."
    
    emb_a = embedder.encode(text_a).tolist()
    emb_b = embedder.encode(text_b).tolist()

    node_a = {
        "text": text_a,
        "embedding": emb_a,
        "metadata": {"type": "secret_doc"}
    }
    node_b = {
        "text": text_b,
        "embedding": emb_b,
        "metadata": {"type": "server_info"}
    }

    # 2. 노드 업서트 (Upsert)
    print("   1️⃣ 노드 A, B 업서트 중...")
    
    try:
        # Node A 요청
        resp_a = requests.post(f"{SERVER_URL}/upsert", json=node_a)
        if not resp_a.ok:
            print(f"❌ Node A 업서트 실패: {resp_a.text}")
            return
        res_a = resp_a.json()
        
        # Node B 요청
        resp_b = requests.post(f"{SERVER_URL}/upsert", json=node_b)
        if not resp_b.ok:
            print(f"❌ Node B 업서트 실패: {resp_b.text}")
            return
        res_b = resp_b.json()

        # ★ [수정] 'new_id' 키 추가
        id_a = res_a.get('new_id') or res_a.get('node_id') or res_a.get('id')
        id_b = res_b.get('new_id') or res_b.get('node_id') or res_b.get('id')

        if id_a is None or id_b is None:
            print(f"❌ 오류: ID 추출 실패. 응답: {res_a}, {res_b}")
            return

        print(f"      - Node A ID: {id_a}")
        print(f"      - Node B ID: {id_b}")

        # 3. 엣지 강제 연결
        print("   2️⃣ 엣지 연결 (Node A <-> Node B)...")
        edge_data = {
            "src": id_a,
            "dst": id_b,
            "weight": 1.0,
            "type": "manual_test"
        }
        # 양방향 연결
        requests.post(f"{SERVER_URL}/add_edge", json=edge_data)
        edge_data["src"], edge_data["dst"] = edge_data["dst"], edge_data["src"]
        requests.post(f"{SERVER_URL}/add_edge", json=edge_data)

        print("\n✅ 데이터 주입 및 연결 완료!")
        print("="*50)
        print("이제 챗봇(test_chat_llm.py)을 켜고 질문하세요:")
        print("👉 질문: \"블랙 펄 프로젝트의 암호가 뭐야?\"")
        print("="*50)

    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")

if __name__ == "__main__":
    inject_scenario()