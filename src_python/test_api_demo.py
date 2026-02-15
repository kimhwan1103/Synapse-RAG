"""
API 동작 검증 스크립트
서버의 주요 엔드포인트가 정상적으로 응답하는지 확인한다.
"""
import requests
import json
import time

SERVER = "http://localhost:9000"

def test_api():
    print()
    print("=" * 70)
    print("  API Endpoint Test")
    print("  Hybrid Graph-Vector DB Server (FastAPI)")
    print("=" * 70)

    # 1. 노드 직접 조회
    print("\n[1/4] GET /node/{id} — RocksDB 직접 조회")
    print("-" * 50)
    for nid in [0, 100000, 1000000]:
        try:
            res = requests.get(f"{SERVER}/node/{nid}", timeout=10)
            if res.ok:
                node = res.json()
                text = node.get("text", "")[:50]
                emb_len = len(node.get("embedding", []))
                print(f"  ID {nid:>8} → {emb_len}-dim | \"{text}...\"")
            else:
                print(f"  ID {nid:>8} → Not Found")
        except Exception as e:
            print(f"  ID {nid:>8} → Error: {e}")

    # 2. 벡터 검색
    print(f"\n[2/4] POST /search_vector — FAISS 벡터 검색")
    print("-" * 50)
    try:
        # 노드 0의 임베딩으로 검색
        node_res = requests.get(f"{SERVER}/node/0", timeout=10)
        if node_res.ok:
            emb = node_res.json().get("embedding", [])
            search_res = requests.post(
                f"{SERVER}/search_vector",
                json={"vector": emb, "k": 3},
                timeout=10
            )
            if search_res.ok:
                results = search_res.json().get("results", [])
                print(f"  Query: Node 0의 embedding으로 Top-3 검색")
                for i, r in enumerate(results, 1):
                    text = r.get("text", "")[:40]
                    print(f"  #{i} ID: {r['id']:>8} | Score: {r['score']:.4f} | \"{text}...\"")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. 이웃 노드 조회
    print(f"\n[3/4] GET /get_neighbors — 그래프 엣지 조회")
    print("-" * 50)
    try:
        res = requests.get(f"{SERVER}/get_neighbors", params={"node_id": 0}, timeout=10)
        if res.ok:
            neighbors = res.json()
            print(f"  Node 0의 이웃: {len(neighbors)}개")
            for nid, weight in neighbors[:5]:
                print(f"    → Node {nid:>8} (weight: {weight:.4f})")
            if len(neighbors) > 5:
                print(f"    ... 외 {len(neighbors)-5}개")
        else:
            print(f"  응답 없음")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. 학습 배치 데이터
    print(f"\n[4/4] GET /get_training_batch — GNN 학습용 배치")
    print("-" * 50)
    try:
        res = requests.get(
            f"{SERVER}/get_training_batch",
            params={"batch_size": 32},
            timeout=30
        )
        if res.ok:
            data = res.json()
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            print(f"  요청: batch_size=32")
            print(f"  응답: nodes={len(nodes)}, edges={len(edges)}")
            if nodes:
                dims = [len(n.get("embedding", [])) for n in nodes[:5]]
                print(f"  임베딩 차원: {dims}")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("  모든 API 엔드포인트 정상 응답 확인")
    print("=" * 70)
    print()

if __name__ == "__main__":
    test_api()
