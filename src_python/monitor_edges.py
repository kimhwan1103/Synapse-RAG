import requests
import time
import os

SERVER_URL = "http://rag-db-server:9000"

def monitor_live_edges():
    print(f"📡 [Monitor] 지식 그래프의 연결 상태를 실시간 감시합니다...")
    print(f"   (서버: {SERVER_URL})")
    print("="*60)

    seen_edges = set() # 이미 본 엣지는 중복 출력 방지

    while True:
        try:
            # 1. 서버에서 데이터 샘플링 (배치 1000개씩 듬성듬성 확인)
            res = requests.get(f"{SERVER_URL}/get_training_batch", params={"batch_size": 1000}, timeout=20)
            
            if not res.ok:
                print(".", end="", flush=True) # 서버 응답 없으면 점 찍기
                time.sleep(2)
                continue

            data = res.json()
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            if not edges:
                time.sleep(1)
                continue

            # ID -> 텍스트 매핑 사전 (내용 확인용)
            # 텍스트가 너무 길면 30자로 자름
            id_to_text = {
                n['id']: n.get('text', '텍스트 없음').replace('\n', ' ')[:40] 
                for n in nodes
            }

            found_new = False
            for edge in edges:
                # 엣지 고유 키 생성 (작은거-큰거 순서로 정렬해서 방향 무시하고 중복 체크)
                src, dst = sorted((edge['src'], edge['dst']))
                edge_key = f"{src}-{dst}"

                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    found_new = True
                    
                    # 텍스트 복원
                    text_src = id_to_text.get(edge['src'], f"ID:{edge['src']}")
                    text_dst = id_to_text.get(edge['dst'], f"ID:{edge['dst']}")
                    
                    print(f"\n🔗 [New Connection Detected!]")
                    print(f"   🅰️  {text_src}...")
                    print(f"       ↕️   (연결됨)")
                    print(f"   🅱️  {text_dst}...")
            
            # 너무 많이 쌓이면 메모리 비우기
            if len(seen_edges) > 10000:
                seen_edges.clear()

            if not found_new:
                # 새로운 게 없으면 잠시 대기
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 모니터링 종료.")
            break
        except Exception as e:
            print(f"⚠️ 에러: {e}")
            time.sleep(2)

if __name__ == "__main__":
    monitor_live_edges()