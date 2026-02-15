import requests
import torch
import os
import concurrent.futures
from tqdm import tqdm

# ==========================================================
# ⚙️ 설정
# ==========================================================
SERVER_URL = "http://rag-db-server:9000"
OUTPUT_FILE = "train_snapshot.pt"
NUM_WORKERS = 16   # 일꾼 수
SCAN_LIMIT = 25000 # 데이터 개수에 맞춰 조절 (약 2만 개 예상)

def export_data_final():
    print(f"📡 서버({SERVER_URL})에서 데이터 추출 시작...")
    
    valid_nodes = []
    print(f"🚀 {NUM_WORKERS}개의 스레드로 ID 0 ~ {SCAN_LIMIT} 스캔 중...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # ★ [수정] URL 경로를 '/get_node/' -> '/node/'로 변경
        futures = {executor.submit(requests.get, f"{SERVER_URL}/node/{i}", timeout=2): i for i in range(SCAN_LIMIT)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=SCAN_LIMIT, desc="Scanning", unit="node"):
            try:
                res = future.result()
                if res.ok:
                    data = res.json()
                    # 유효성 검사
                    if data and "embedding" in data and len(data["embedding"]) == 768:
                        valid_nodes.append(data)
            except:
                pass

    print(f"✅ 다운로드 완료! 유효 데이터: {len(valid_nodes)}개")

    if len(valid_nodes) == 0:
        print("⚠️ 데이터가 0개입니다. 주소가 맞는데도 0개라면 'inject_test_data.py'를 먼저 실행하세요!")
        return

    # 파일 저장
    snapshot = {
        "nodes": valid_nodes
    }
    torch.save(snapshot, OUTPUT_FILE)
    print(f"💾 스냅샷 저장 완료: {os.path.abspath(OUTPUT_FILE)}")
    print("👉 이제 'python train_llm_projector.py'를 실행하세요!")

if __name__ == "__main__":
    export_data_final()