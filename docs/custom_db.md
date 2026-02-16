# Hybrid Graph-Vector Database

> RocksDB + FAISS + LRU Cache를 결합한 C++ 기반 커스텀 하이브리드 데이터베이스

## Overview

상용 그래프 DB나 벡터 DB에 의존하지 않고, 본 프로젝트의 요구사항(768차원 벡터 검색 + 그래프 탐색 + 텍스트 저장)에 최적화된 저장소 엔진을 C++로 직접 구현했다.

```
┌─────────────────────────────────────────────────┐
│           FastAPI Server (db_server_v2.py)       │
│         REST API + FAISS Index (768-dim)         │
├─────────────────────────────────────────────────┤
│           C++ Core (hybrid_graphDB_v2)           │
│    ┌──────────────┐    ┌──────────────────┐     │
│    │  LRU Cache   │───→│    RocksDB       │     │
│    │  (2000 nodes)│    │  (Persistent)    │     │
│    └──────────────┘    └──────────────────┘     │
└─────────────────────────────────────────────────┘
```

## Data Model

### HybridNode

그래프와 벡터를 하나의 노드 단위로 통합한 핵심 데이터 구조이다.

```cpp
struct HybridNode {
    int id;                        // 고유 식별자
    std::string text_content;      // 원본 텍스트 (위키피디아 문서, 대화 등)
    std::vector<float> embedding;  // 768차원 벡터 임베딩
    std::vector<int> neighbors;    // 인접 노드 ID 목록
};
```

### Storage Layout (RocksDB)

| Key Prefix | Format | Description |
|------------|--------|-------------|
| `N_{id}` | `id\|text\|emb_size\|emb_values\|nb_size\|nb_ids` | 노드 데이터 |
| `E_{src_id}` | `dst1:weight1,dst2:weight2,...` | 엣지 데이터 |

파이프(`|`) 구분자 기반의 경량 직렬화 포맷을 사용하여, JSON/Protobuf 대비 파싱 오버헤드를 최소화했다.

## Architecture

### 2-Tier Caching

```
GET /node/{id}
    │
    ▼
┌──────────┐  hit   ┌───────────────┐
│ LRU Cache│───────→│ Return Node   │
│ (Memory) │        └───────────────┘
└────┬─────┘
     │ miss
     ▼
┌──────────┐  found  ┌───────────────┐
│ RocksDB  │────────→│ Update Cache  │
│ (Disk)   │         │ + Return Node │
└──────────┘         └───────────────┘
```

- **Tier 1 (LRU Cache)**: 최대 2,000개 노드를 메모리에 유지. 이중 연결 리스트 + 해시맵으로 O(1) 접근
- **Tier 2 (RocksDB)**: LSM-tree 기반 디스크 영속 저장소. 캐시 미스 시 로드 후 LRU에 승격

### Vector Search (FAISS)

- **Index Type**: `IndexFlatIP` (Inner Product)
- **Dimension**: 768 (ko-sroberta-multitask 출력 차원)
- **검색 방식**: L2 정규화된 벡터의 내적 = 코사인 유사도와 동일

```python
# 서버 시작 시 FAISS 인덱스 구축
faiss.normalize_L2(all_embeddings)
index = faiss.IndexFlatIP(768)
index.add(all_embeddings)
```

## C++ Implementation

### V1 (`src_cpp/`)

초기 구현. pybind11로 Python 바인딩을 제공한다.

```
src_cpp/
├── hybrid_graphDB.hpp    # 클래스 선언 (HybridNode, HybridGraphDB)
├── hybrid_graphDB.cpp    # 핵심 구현 (700+ lines)
├── bindings.cpp          # pybind11 바인딩
└── setup.py              # 빌드 설정
```

**주요 기능:**
- `add_node()` / `get_node()`: 노드 CRUD
- `add_edge()` / `get_neighbors()`: 그래프 탐색
- `build_similarity_graph(threshold)`: 코사인 유사도 기반 자동 엣지 생성
- `build_knn_graph(k)`: K-최근접 이웃 그래프 구축
- `get_similarity(id1, id2)`: 두 노드 간 코사인 유사도 계산

### V2 (`src_server/src_db/`)

운영 환경용 개선 버전. CMake 빌드 시스템을 사용한다.

```
src_server/src_db/
├── hybrid_graphDB_v2.hpp     # 클래스 선언
├── hybrid_graphDB_v2.cpp     # 핵심 구현 (630 lines)
├── python_bindings.cpp       # pybind11 바인딩
├── db_maintenance_tool.cpp   # CLI 유지보수 도구
├── CMakeLists.txt            # CMake 빌드
└── setup.py                  # Python 패키지 빌드
```

**V2 추가 기능:**
- `is_text_complete()`: 텍스트 완결성 검증 (불완전한 문장 탐지)
- `clean_and_validate_text()`: 텍스트 정제 (최소 길이, 잘린 문장 제거)
- `validate_text_quality()`: 전체 노드 품질 검사
- `find_orphan_nodes()`: 엣지 없는 고립 노드 탐색
- `print_graph_stats()`: 그래프 통계 출력

### Build

```bash
# V1 (setuptools)
cd src_cpp && pip install .

# V2 (CMake)
cd src_server/src_db
mkdir build && cd build
cmake .. && make -j$(nproc)

# 또는 Python 패키지로
cd src_server/src_db && pip install .
```

## API Endpoints

FastAPI 서버(`db_server_v2.py`)가 REST API를 제공한다.

### Node Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upsert` | POST | 노드 추가 (유사도 0.92 이상이면 중복으로 판단, 기존 ID 반환) |
| `/node/{id}` | GET | 단일 노드 조회 (텍스트 + 임베딩) |
| `/get_embeddings_bulk` | GET | 랜덤 노드 임베딩 벌크 조회 (학습용) |
| `/count` | GET | 총 노드 수 |
| `/stats` | GET | DB 통계 |

### Vector Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search_vector` | POST | FAISS 벡터 유사도 검색 (top-k) |

```json
// Request
{ "vector": [768 floats], "k": 5 }

// Response
{
  "status": "ok",
  "results": [
    { "id": 1001, "text": "...", "embedding": [...], "score": 0.89 }
  ]
}
```

### Graph Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/add_edge` | POST | 단일 엣지 추가 |
| `/add_edges_batch` | POST | 배치 엣지 추가 |
| `/get_neighbors` | GET | 인접 노드 조회 |
| `/get_training_batch` | GET | GNN 학습용 서브그래프 배치 |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cleanup_memory` | POST | LRU 캐시 초기화 + GC + malloc trim |

## Performance

| Operation | Complexity | Note |
|-----------|-----------|------|
| `get_node()` (캐시 히트) | O(1) | LRU 해시맵 조회 |
| `get_node()` (캐시 미스) | O(log n) | RocksDB LSM-tree 조회 |
| `add_node()` | O(log n) | RocksDB 쓰기 + LRU 업데이트 |
| `add_edge()` | O(1) | 문자열 append |
| `get_neighbors()` | O(k) | k = 이웃 수 |
| `search_vector()` | O(n) | FAISS FlatIP (brute-force) |

## Data Pipeline

위키피디아 한국어 덤프를 파싱하여 DB에 적재하는 파이프라인이다.

```
kowiki-YYYYMMDD-pages-articles-multistream.xml.bz2
    │
    ▼  wiki_dump_parser.py
    │  - XML 스트리밍 파싱 (lxml.etree.iterparse)
    │  - 문서 추출 + 텍스트 정제
    │
    ▼  wiki_full_pipeline.py
    │  - ko-sroberta-multitask로 768차원 임베딩 생성
    │  - /upsert API로 DB 적재
    │  - 코사인 유사도 ≥ 0.75 기준 자동 엣지 생성
    │
    ▼  FAISS 인덱스 구축 (서버 시작 시)
```

## Known Issues: Index Misalignment

2.45M 노드 규모에서 FAISS 인덱스와 RocksDB 간 ID 매핑 불일치 현상이 발생했다. 자세한 분석은 [메인 README의 Post-Mortem](../README.md#post-mortem-index-misalignment) 섹션을 참조한다.

**요약:**
- RocksDB: 2,454,054개 vs FAISS: 2,454,052개 (2개 Ghost Data)
- 원인: 3개 독립 저장소(RocksDB, FAISS, Python List) 간 트랜잭션 부재
- 교훈: 인프라 레이어의 정합성 문제는 애플리케이션 코드가 아닌 검증된 저장소 엔진에 위임해야 한다

## File Map

```
src_server/
├── db_server_v2.py         # FastAPI 서버 (REST API + FAISS)
├── metadata_manager.py     # 메타데이터 관리 (node_count 캐싱)
├── wiki_dump_parser.py     # Wikipedia XML 파서
├── wiki_full_pipeline.py   # 데이터 적재 파이프라인
├── Dockerfile
└── src_db/                 # C++ DB V2
    ├── hybrid_graphDB_v2.hpp
    ├── hybrid_graphDB_v2.cpp
    ├── python_bindings.cpp
    ├── db_maintenance_tool.cpp
    ├── CMakeLists.txt
    └── setup.py

src_cpp/                    # C++ DB V1
├── hybrid_graphDB.hpp
├── hybrid_graphDB.cpp
├── bindings.cpp
└── setup.py
```
