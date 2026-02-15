import json
import os
import logging

# 데이터 저장 경로 (Docker 볼륨 경로와 일치해야 함)
DATA_DIR = "/app/data/my_conversation_db"
META_FILE = os.path.join(DATA_DIR, "meta.json")

logger = logging.getLogger("BrainServer")

def load_node_count(db=None):
    """
    서버 시작 시 노드 개수를 불러옵니다.
    ★ C++ 바인딩(db.get_node_count)이 깨져도 작동하도록, 
       철저하게 'meta.json' 파일에 의존합니다.
    """
    # 1. 폴더가 없으면 생성
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
        except Exception:
            pass

    # 2. 메타데이터 파일 읽기 (Python Native)
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r") as f:
                data = json.load(f)
                count = data.get("node_count", 0)
                logger.info(f"[Meta] ✅ Found metadata! Loaded count: {count}")
                return count
        except Exception as e:
            logger.error(f"[Meta] Failed to read metadata: {e}")
            return 0

    # 3. 파일이 없으면? (Cold Start)
    # 기존에는 여기서 db.get_node_count()를 호출했지만, 
    # C++ 바인딩 문제로 에러가 나므로 과감히 생략하고 0을 리턴합니다.
    logger.warning("[Meta] No metadata file found. Assuming new DB (Count: 0).")
    
    # 혹시라도 DB는 있는데 메타파일만 지워진 경우를 대비해
    # 안전하게 체크만 해봅니다 (없으면 말고 식)
    if db is not None and hasattr(db, "get_node_count"):
        try:
            return db.get_node_count()
        except:
            pass
            
    return 0

def save_node_count(count):
    """
    노드가 추가될 때마다(혹은 주기적으로) 개수를 파일에 저장합니다.
    """
    try:
        with open(META_FILE, "w") as f:
            json.dump({"node_count": count}, f)
        # logger.info(f"[Meta] Saved count: {count}") # 너무 자주 찍히면 주석 처리
    except Exception as e:
        logger.error(f"[Meta] Failed to save metadata: {e}")