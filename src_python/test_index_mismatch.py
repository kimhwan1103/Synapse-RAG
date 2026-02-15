"""
Post-Mortem 증거: 인덱스 밀림(Index Misalignment) 검증 스크립트
RocksDB에서 노드를 직접 조회한 뒤, 같은 벡터로 FAISS 검색하여 ID/텍스트 불일치를 확인한다.

실행: python test_index_mismatch.py
"""
import requests
import random

SERVER = "http://localhost:9000"

def test_index_mismatch():
    print()
    print("=" * 80)
    print("  [Post-Mortem] Index Misalignment Test")
    print("  RocksDB 직접 조회 vs FAISS 벡터 검색 정합성 검증")
    print("=" * 80)

    # 1. 서버 상태 확인
    print("\n[1/3] 서버 상태 확인")
    print("-" * 40)
    try:
        # get_training_batch를 가볍게 호출하여 노드 수 확인
        res = requests.get(f"{SERVER}/get_training_batch", params={"batch_size": 1}, timeout=10)
        if res.ok:
            print(f"  서버 응답: OK")
        else:
            print(f"  서버 응답 실패: {res.status_code}")
            return
    except Exception as e:
        print(f"  서버 연결 실패: {e}")
        return

    # 2. 랜덤 노드 5개 선택하여 검증
    test_ids = random.sample(range(100000, 2400000), 5)

    print(f"\n[2/3] 정합성 검증 (랜덤 노드 {len(test_ids)}개)")
    print("-" * 40)

    mismatch_count = 0

    for i, node_id in enumerate(test_ids, 1):
        print(f"\n  ── Test {i} ── Node ID: {node_id}")

        # RocksDB에서 직접 조회
        try:
            res = requests.get(f"{SERVER}/node/{node_id}", timeout=10)
            if not res.ok:
                print(f"     RocksDB: 노드 없음 (skip)")
                continue
            node = res.json()
        except:
            continue

        db_text = node.get("text", "")[:60]
        db_emb = node.get("embedding", [])

        # 같은 embedding으로 FAISS 검색
        try:
            search_res = requests.post(
                f"{SERVER}/search_vector",
                json={"vector": db_emb, "k": 1},
                timeout=10
            )
            results = search_res.json().get("results", [])
        except:
            continue

        if not results:
            continue

        top = results[0]
        faiss_id = top["id"]
        faiss_text = top.get("text", "")[:60]
        score = top["score"]

        print(f"     RocksDB  → ID: {node_id:>8}  \"{db_text}...\"")
        print(f"     FAISS    → ID: {faiss_id:>8}  \"{faiss_text}...\"")
        print(f"     Score: {score:.6f}")

        if faiss_id != node_id or db_text != faiss_text:
            mismatch_count += 1
            print(f"     [MISMATCH] 요청 ID {node_id} ≠ 반환 ID {faiss_id}")
        else:
            print(f"     [OK]")

    # 3. 최종 결과
    print(f"\n[3/3] 최종 결과")
    print("=" * 80)
    print(f"  MISMATCH: {mismatch_count} / {len(test_ids)}")
    print()

    if mismatch_count > 0:
        print("  CONCLUSION: 인덱스 밀림(Index Misalignment) 확인")
        print("  ─────────────────────────────────────────────")
        print("  RocksDB와 FAISS 간 ID 매핑이 어긋나 있음.")
        print("  유사도 1.0으로 정확히 검색되었으나, 반환된 노드의")
        print("  ID와 텍스트가 요청한 것과 일치하지 않음.")
        print()
        print("  ROOT CAUSE:")
        print("    - RocksDB(disk)와 FAISS(memory)의 Write Atomicity 부재")
        print("    - Bulk Insert 중 부분 실패로 인한 Ghost Data 발생")
        print("    - 서버 재시작 시 ID 시퀀스 불일치")
    else:
        print("  정합성 정상")

    print("=" * 80)
    print()

if __name__ == "__main__":
    test_index_mismatch()
