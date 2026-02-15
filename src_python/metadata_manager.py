import json
import os

# ★ 핵심 수정: 이 파일(metadata_manager.py)의 위치를 기준으로 경로를 계산합니다.
# 이렇게 하면 /workspace가 아니어도, 어떤 폴더에서 실행하든 무조건 정확한 위치를 잡습니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
META_FILE = os.path.join(CURRENT_DIR, "../data/my_conversation_db/meta.json")
META_FILE = os.path.abspath(META_FILE)  # 경로 깔끔하게 정리

def save_node_count(count):
    """현재 노드 개수를 JSON 파일에 저장"""
    directory = os.path.dirname(META_FILE)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(META_FILE, 'w') as f:
        json.dump({"total_nodes": count}, f)
    print(f"[Meta] Saved node count to: {META_FILE}")

def load_node_count(db_instance=None):
    """
    1순위: meta.json에서 읽기 (O(1))
    2순위: 파일 없으면 DB에서 세기 (O(N) - 느림)
    """
    # 디버깅: 실제로 어디를 찾고 있는지 눈으로 확인
    print(f"[Meta] Looking for metadata file at: {META_FILE}")

    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, 'r') as f:
                data = json.load(f)
                print(f"[Meta] ✅ Found metadata! Loaded count: {data['total_nodes']}")
                return data["total_nodes"]
        except Exception as e:
            print(f"[Meta] File read error: {e}")
    else:
        print(f"[Meta] ❌ File NOT found.")
    
    # 파일이 없으면 어쩔 수 없이 DB에서 세야 함
    if db_instance:
        print("[Meta] Counting nodes from DB (This will take time)...")
        count = db_instance.get_node_count()
        save_node_count(count)
        return count
    
    return 0