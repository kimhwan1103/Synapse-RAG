import requests
import json

SERVER_URL = "http://rag-db-server:9000"

def check_connections():
    print(f"📡 DB 서버({SERVER_URL})에 연결 중...")
    
    try:
        # 배치 사이즈를 100개만 가져와서 뜯어봅니다.
        res = requests.get(f"{SERVER_URL}/get_training_batch", params={"batch_size": 100}, timeout=10)
        
        if not res.ok:
            print(f"❌ 서버 응답 오류: {res.status_code}")
            return

        data = res.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print("\n" + "="*40)
        print(f"✅ 데이터 샘플 확보 완료")
        print(f"   - 가져온 노드 수: {len(nodes)}")
        print(f"   - 가져온 엣지 수: {len(edges)}")
        print("="*40)

        if len(edges) == 0:
            print("⚠️ 경고: 이 샘플에는 엣지가 하나도 없습니다. (운이 나빴거나 데이터가 희소함)")
        else:
            print("\n🔍 연결 관계 샘플 (상위 5개):")
            
            # ID를 텍스트로 바꾸기 위한 맵핑 (텍스트가 있다면)
            id_to_text = {n['id']: n.get('text', '텍스트 없음')[:20] for n in nodes}
            
            for i, edge in enumerate(edges[:5]):
                src_id = edge['src']
                dst_id = edge['dst']
                
                src_text = id_to_text.get(src_id, f"ID:{src_id}")
                dst_text = id_to_text.get(dst_id, f"ID:{dst_id}")
                
                print(f"   🔗 [연결 {i+1}] {src_text} ... <---> ... {dst_text}")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    check_connections()