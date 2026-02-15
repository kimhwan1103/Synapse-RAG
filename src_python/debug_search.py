import requests
import numpy as np
# 임베딩을 만들기 위해 필요 (기존 코드에서 사용한 임베딩 모델 로드)
from sentence_transformers import SentenceTransformer

# 모델 로드 (서버랑 똑같은 거여야 함)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def debug_search(text):
    print(f"🔎 검색어: '{text}'")
    vector = model.encode(text).tolist()
    
    try:
        res = requests.post(
            "http://rag-db-server:9000/search_vector", 
            json={"vector": vector, "k": 3},
            timeout=5
        )
        results = res.json().get("results", [])
        
        for idx, item in enumerate(results):
            print(f"   [{idx+1}등] Score: {item['score']:.4f} | Text: {item['text'][:50]}...")
            
    except Exception as e:
        print(f"❌ 에러: {e}")

if __name__ == "__main__":
    debug_search("GNN")
    debug_search("대한민국")