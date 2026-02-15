import requests
from sentence_transformers import SentenceTransformer

# 1. 모델 로드 (서버와 동일한 모델 사용 필수)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def test_sanity_check():
    print("🧪 [Sanity Check] 서버 상태 검증 시작...")
    
    # 1. 테스트용 "확실한" 데이터 생성
    test_text = "테슬라는 일론 머스크가 만든 전기차 회사이다."
    test_vec = model.encode(test_text).tolist()
    
    # 2. 데이터 주입 (★수정됨: /add -> /upsert)
    try:
        print("   ➡️ 데이터 주입 시도...", end=" ")
        
        # 서버 코드(db_server_v2.py)에 정의된 엔드포인트는 /upsert 입니다.
        res = requests.post(
            "http://rag-db-server:9000/upsert",
            json={"text": test_text, "embedding": test_vec},
            timeout=5
        )
        
        if res.ok:
            data = res.json()
            # upsert는 'new_id' 또는 이미 존재할 경우 'id'를 반환합니다.
            new_id = data.get("new_id") if data.get("status") == "created" else data.get("id")
            print(f"성공! (ID: {new_id}, Status: {data.get('status')})")
        else:
            print(f"실패 (Status: {res.status_code})")
            print(f"응답 내용: {res.text}")
            return
            
        # 3. 바로 검색 시도 (같은 벡터로 검색)
        print("   🔍 방금 넣은 데이터 검색 시도...", end=" ")
        search_res = requests.post(
            "http://rag-db-server:9000/search_vector",
            json={"vector": test_vec, "k": 1},
            timeout=5
        )
        
        results = search_res.json().get("results", [])
        if not results:
            print("❌ 결과 없음 (FAISS 인덱스 갱신 안됨?)")
            return
            
        top_result = results[0]
        print(f"\n   [결과] ID: {top_result['id']} | Score: {top_result['score']:.4f}")
        print(f"   [내용] {top_result['text']}")
        
        # 4. 판정
        # 공백이나 줄바꿈 차이 무시하고 비교
        if top_result['text'].strip() == test_text.strip():
            print("\n✅ 결론: 서버 코드는 정상입니다!") 
            print("   (만약 기존 검색 결과가 이상했다면, 기존에 적재된 위키 데이터의 ID 매핑이 꼬인 것입니다.)")
            print("👉 조치: DB 폴더(data/)를 삭제하고 서버를 재시작하여 데이터를 다시 적재하세요.")
        else:
            print("\n❌ 결론: 서버 코드(인덱싱 로직)에 버그가 있습니다.")
            print(f"👉 원인: 넣은 ID는 {new_id}인데, 검색된 ID는 {top_result['id']}이며 내용이 다릅니다.")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")

if __name__ == "__main__":
    test_sanity_check()