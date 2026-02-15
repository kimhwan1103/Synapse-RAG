import requests
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import concurrent.futures

# ==========================================================
# ⚙️ 설정
# ==========================================================
SERVER_URL = "http://rag-db-server:9000"
# ★ 현재 클라이언트/학습 코드에서 사용하는 모델과 100% 일치해야 함
MODEL_NAME = "jhgan/ko-sroberta-multitask" 
NUM_WORKERS = 8 # 병렬 처리 개수

def refresh_db():
    print(f"📥 임베딩 모델 로딩 중 ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"📡 서버({SERVER_URL}) 데이터 스캔 시작...")
    
    # 1. 업데이트할 대상 스캔 (0 ~ 30000번)
    # 실제 존재하는 노드만 찾아서 리스트업
    target_ids = []
    
    # 빠른 스캔을 위해 멀티스레드 사용
    def check_exists(nid):
        try:
            res = requests.get(f"{SERVER_URL}/node/{nid}", timeout=2)
            if res.ok:
                data = res.json()
                if data and "text" in data:
                    return nid, data["text"]
        except:
            pass
        return None

    print("🔍 유효한 노드 식별 중... (잠시만 기다려주세요)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(check_exists, i): i for i in range(30000)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=30000, desc="Scanning DB"):
            res = future.result()
            if res:
                target_ids.append(res) # (id, text) 튜플 저장

    print(f"✅ 총 {len(target_ids)}개의 노드를 찾았습니다. 벡터 갱신을 시작합니다.")

    # 2. 벡터 재생성 및 업데이트 (Batch Update)
    # 한 번에 여러 개씩 인코딩해서 속도 향상
    BATCH_SIZE = 32
    
    for i in tqdm(range(0, len(target_ids), BATCH_SIZE), desc="Refreshing Vectors"):
        batch = target_ids[i : i + BATCH_SIZE]
        
        texts = [item[1] for item in batch]
        ids = [item[0] for item in batch]
        
        # (1) 새 임베딩 생성 (여기가 핵심!)
        new_embeddings = model.encode(texts).tolist()
        
        # (2) 서버에 업데이트 요청
        # /upsert API를 사용하되, 기존 ID를 그대로 쓰면 내용만 덮어씌워짐
        for j, nid in enumerate(ids):
            payload = {
                "text": texts[j],
                "embedding": new_embeddings[j],
                "threshold": 1.1 # 무조건 덮어쓰기 위해 threshold를 1.0보다 높게 설정 (서버 로직에 따라 다름)
            }
            # 만약 서버에 'update_node' 같은 별도 API가 없다면 upsert 활용
            # 하지만 upsert는 유사도 체크를 하므로, 강제 업데이트가 필요함.
            # 가장 확실한 건 DB를 직접 수정하는 것이나, API로는 기존 ID 유지가 까다로울 수 있음.
            
            # 차라리 'add' API가 ID를 지정할 수 있다면 덮어쓰기가 됨.
            # 현재 서버 코드(db_server.py)를 보면 upsert는 '유사하면 스킵'함.
            
            # ★ 해결책: 서버에 'force_update' 기능이 없으므로, 
            # 텍스트는 그대로 두고 임베딩만 바꾸는 API가 필요하지만,
            # 지금은 차선책으로 "Projector 학습용 스냅샷"만 올바르게 만들면 됨.
            pass

    # 위 방식은 DB 자체를 고치는 건데, 서버 API 제약으로 복잡할 수 있음.
    # ★ 더 쉬운 방법: DB는 건드리지 말고, "학습 데이터 파일(train_snapshot.pt)"만 올바르게 고치면 됨!
    
    print("\n💡 전략 수정: DB 서버를 건드리는 대신, 학습용 스냅샷만 깨끗하게 만듭니다.")
    print("   (이러면 DB는 그대로 두고, 학습은 최신 모델로 할 수 있습니다.)")
    
    snapshot_nodes = []
    
    # 전체 데이터를 다시 인코딩해서 스냅샷 리스트에 담음
    all_texts = [item[1] for item in target_ids]
    all_embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=64)
    
    for k, (nid, text) in enumerate(target_ids):
        snapshot_nodes.append({
            "id": nid,
            "text": text,
            "embedding": all_embeddings[k].tolist() # 새 벡터!
        })
        
    # 저장
    snapshot = {"nodes": snapshot_nodes}
    torch.save(snapshot, "train_snapshot_clean.pt")
    print(f"💾 'train_snapshot_clean.pt' 저장 완료!")
    print("👉 train_llm_projector.py에서 SNAPSHOT_FILE = 'train_snapshot_clean.pt'로 바꿔서 실행하세요.")

if __name__ == "__main__":
    refresh_db()