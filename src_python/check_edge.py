import requests
import json
import time

SERVER_URL = "http://rag-db-server:9000"

def force_find_edges():
    print(f"📡 DB 서버({SERVER_URL})에 '대형 그물'을 던집니다... (최대 30초 소요)")
    
    try:
        # ★ 핵심: 배치 사이즈를 2000으로 대폭 늘림 (확률 100% 도전)
        res = requests.get(
            f"{SERVER_URL}/get_training_batch", 
            params={"batch_size": 2000}, 
            timeout=60 # 데이터가 많으니 타임아웃도 60초로
        )
        
        if not res.ok:
            print(f"❌ 요청 실패 (Status: {res.status_code})")
            return

        data = res.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print("\n" + "="*60)
        print(f"📊 수확 결과:")
        print(f"   - 노드: {len(nodes)}개")
        print(f"   - 엣지: {len(edges)}개")
        print("="*60)

        if len(edges) > 0:
            print(f"🎉 찾았습니다! {len(edges)}개의 연결을 확인했습니다.")
            
            # 텍스트 매핑
            id_to_text = {n['id']: n.get('text', '텍스트 없음') for n in nodes}
            
            # 3개만 샘플 출력
            print("\n🔍 연결 내용 미리보기:")
            for i, edge in enumerate(edges[:3]):
                src = id_to_text.get(edge['src'], str(edge['src']))[:30]
                dst = id_to_text.get(edge['dst'], str(edge['dst']))[:30]
                print(f"   🔗 {src}... <---> {dst}...")
        else:
            print("😱 2000개를 뽑았는데도 없다면, DB 초기화가 필요할 수도 있습니다.")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    force_find_edges()