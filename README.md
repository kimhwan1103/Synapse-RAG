# Synapse-RAG: GNN-LLM Joint Learning 기반의 지식 그래프 주입형 RAG 시스템

GNN 기반 그래프 추론과 벡터 검색을 결합한 하이브리드 데이터베이스 위에서 동작하는 RAG(Retrieval-Augmented Generation) 시스템

## Motivation

- **기존 RAG의 한계**: 기존 RAG(Retrieval-Augmented Generation)는 문장을 벡터로 변환한 뒤 코사인 유사도만으로 검색한다.
    - 문제점: "테슬라"를 검색하면 '자동차'와 '니콜라 테슬라'가 혼재되어 나오고, 그 사이의 인과 관계나 연관성은 파편화된다.

- **본 프로젝트의 접근**: 인간의 뇌처럼 지식을 구조적(Graph)으로 저장하고 탐색한다면, LLM이 훨씬 더 깊이 있고 논리적인 답변을 할 수 있을 것이다.

- **본 프로젝트의 목표**: 한국어 위키피디아의 지식을 지식 그래프로 구축하고, 이를 GNN으로 압축하여 LLM에 주입하는 검색 시스템을 만든다.

## 핵심 아키텍처 컨셉

본 프로젝트는 크게 3가지 핵심 모듈로 구성되었다.

### 1. 자체 구축한 하이브리드 지식 엔진

> "상용 DB에 의존하지 않고, 그래프 데이터 처리에 최적화된 엔진을 직접 만든다."

- **역할**: 대규모 그래프 데이터를 저장하고 검색하는 핵심 저장소
- **기술**:
    - RocksDB: 텍스트와 메타데이터를 저장하는 Key-Value 저장소
    - FAISS: 768차원 벡터를 초고속으로 검색하는 인메모리 인덱스

### 2. Gardener: GNN-based Retriever

> "단순히 찾는 것이 아니라, 관계를 학습한다."

- **역할**: 그래프의 위상학적 구조를 학습하는 GNN 모델
- **알고리즘**: GraphSAGE + GAE(Graph Auto-Encoder)
- **기능**:
    - 단순히 키워드가 비슷한 문서를 찾는 것이 아니라, 해당 문서와 연결된 다른 지식까지 함께 고려하여 임베딩을 생성한다.
    - 마치 정원사(Gardener)가 가지치기를 하듯, 의미 있는 연결을 강화하고 노이즈를 제거한다.

### 3. Projector: Graph Adapter

> "텍스트로 변환하지 않고, 지식을 LLM에 직접 주입한다."

- **핵심 차별점**:
    - 일반적인 RAG는 검색된 정보를 다시 텍스트로 변환하여 프롬프트에 삽입한다.
    - 본 프로젝트는 GNN이 압축한 벡터를 선형 변환하여 LLM의 임베딩 공간에 직접 주입한다.
    - **효과**: 텍스트로 표현할 수 없는 구조적 맥락과 관계 정보까지 전달 가능

## 기대 효과

- **환각 감소**: LLM이 모르는 내용을 지어내는 대신, GNN이 제공한 구조적 근거를 바탕으로 답변한다.
- **추론 능력 향상**: 단순 정보 나열이 아니라, A와 B가 왜 연결되었는지(인과관계)를 이해하는 답변을 생성한다.
- **효율성**: 텍스트를 프롬프트에 삽입하는 것보다 압축된 벡터를 사용하므로, 토큰 비용 절감 및 처리 속도가 향상된다.


## System Architecture

```
Query Text
    │
    ▼
┌──────────────────┐
│  Embedding Model  │  (jhgan/ko-sroberta-multitask, 768-dim)
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────────────────────┐
│ FAISS  │ │   Hybrid Graph DB (C++)     │
│ Vector │ │  ┌───────────┐              │
│ Search │ │  │ LRU Cache │ ← 2000 nodes│
│        │ │  └─────┬─────┘              │
│        │ │        ▼                    │
│        │ │  ┌───────────┐              │
│        │ │  │  RocksDB  │ ← Persistent│
│        │ │  └───────────┘              │
└───┬────┘ └──────────┬──────────────────┘
    │                 │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   GNN Encoder   │  (GCN / GraphSAGE)
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │    Projector     │  (GNN → LLM Bridge)
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   Qwen-2.5-1.5B │  (Answer Generation)
    └─────────────────┘
```

## Model Architecture

### SharedGCNEncoder (공유 백본)

Gardener와 HybridQueryGNN이 동일한 가중치를 공유하는 GCN 인코더이다. 정보 손실을 방지하기 위해 내부 히든 차원을 입력보다 크게 설정했다.

```
Input (768-dim)
  → GCNConv (768 → 1024) → ReLU
  → GCNConv (1024 → 768)
Output (768-dim)
```

### HybridQueryGNN (추론 엔진)

사용자의 질문(query)과 서브그래프를 함께 받아, 그래프 구조를 반영한 컨텍스트 벡터를 생성한다. Local(노드 레벨)과 Global(그래프 레벨) 두 경로로 나뉜다.

```
[Node Embeddings] ──→ SharedGCNEncoder ──→ h_local (768-dim)
                                              │
[Query Embedding] ──→ Linear Projection ──→ h_global (768-dim)
                                              │
                      Scaled Dot-Product Attention
                      (h_global × h_local / √d)
                                              │
                      Weighted Sum ──→ global_message
                                              │
                      GRU Update ──→ h_global_new
                                              │
                      Broadcast + Residual ──→ h_final (per node)
```

- **Local 경로**: SharedGCNEncoder가 이웃 노드의 정보를 집계하여 노드별 표현을 생성한다.
- **Global 경로**: 질문 벡터를 초기 상태로 하여, Attention으로 중요 노드를 선별한 뒤 GRU로 글로벌 컨텍스트를 갱신한다.
- **출력**: 노드별 임베딩(`h_final`) + 그래프 요약 벡터(`h_global_new`) + Attention 가중치

### GraphSAGE Encoder (Gardener 전용)

Gardener가 사용하는 인코더로, 768차원 임베딩을 128차원 잠재 공간으로 압축한다. 이 잠재 공간에서 FAISS 검색을 수행하여 새로운 엣지를 발견한다.

```
Input (768-dim)
  → Linear (768 → 256) → BatchNorm → ELU → Dropout
  → SAGEConv (256 → 256) → BatchNorm → ELU → Dropout
  → SAGEConv (256 → 128)
  → L2 Normalize × 3.0 (Max Norm Scaling)
Output (128-dim)
```

- Graph Auto-Encoder(GAE)의 인코더로 사용되며, 링크 예측(Reconstruction Loss)으로 학습된다.
- 학습된 128차원 벡터로 FAISS 인덱스를 구축하여, 원본 768차원에서 발견하지 못한 의미적 연결을 탐색한다.

### GraphLLM_Connector (Projector 학습 모델)

GNN의 출력을 LLM의 입력 공간으로 변환하는 브릿지이다. GNN은 동결(frozen)하고, Projector와 LLM의 LoRA 어댑터만 학습한다.

```
[Subgraph] ──→ HybridQueryGNN (frozen) ──→ h_global (768-dim)
                                               │
                    Projector: Linear (768 → LLM_hidden)
                               → LayerNorm → GELU
                               → Linear (LLM_hidden → LLM_hidden)
                                               │
                                         graph_token (1 token)
                                               │
              ┌────────────────────────────────┘
              ▼
  [graph_token] + [text_input_embeddings]  ──→  Qwen-2.5-1.5B (LoRA)
                                                     │
                                               Answer Tokens
```

- LLM: Qwen-2.5-1.5B-Instruct (4-bit 양자화, LoRA rank=8, alpha=32, target: q_proj, v_proj)
- **핵심**: `graph_token`이 텍스트 임베딩 시퀀스의 맨 앞에 결합(concatenate)되어, LLM이 그래프 컨텍스트를 조건으로 답변을 생성한다.

### GraphRAG_Generator (Full Model)

GNN 인코더와 Transformer 디코더를 결합한 독립형 생성 모델이다. LLM 없이 자체적으로 텍스트를 생성한다.

```
[Subgraph] ──→ HybridQueryGNN ──→ h_global + h_nodes
                                        │
                    Memory = [h_global ; h_nodes_dense]
                                        │
[Target Tokens] ──→ Embedding → Positional Encoding
                                        │
                    Transformer Decoder (3 layers, 4 heads)
                    (Causal Mask + Memory Key Padding Mask)
                                        │
                    Linear → Vocab Logits (30,522 tokens)
```

- 디코더는 BERT 토크나이저(bert-base-multilingual-cased, vocab 30,522)를 사용한다.
- 200스텝마다 Gardener의 최신 가중치를 동기화(Brain Sync)하여, 지속적으로 개선되는 GNN 인코더를 반영한다.

## Directory Structure

```
Synapse-RAG/
├── src_cpp/                # C++ 코어 DB 엔진
│   ├── hybrid_graphDB.hpp  #   HybridNode & HybridGraphDB 클래스 정의
│   ├── hybrid_graphDB.cpp  #   RocksDB 연동, LRU 캐시, 직렬화
│   ├── bindings.cpp        #   pybind11 Python 바인딩
│   └── setup.py            #   빌드 설정
│
├── src_server/             # FastAPI 백엔드 서버
│   ├── db_server_v2.py     #   REST API 엔드포인트
│   ├── metadata_manager.py #   메타데이터 관리
│   ├── src_db/             #   C++ DB v2 (CMake 빌드)
│   ├── wiki_full_pipeline.py   # Wikipedia 데이터 파이프라인
│   └── wiki_dump_parser.py     # XML 덤프 파서
│
├── src_python/             # GNN 모델 & 학습
│   ├── models.py           #   HybridQueryGNN, SharedGCNEncoder
│   ├── model_llm.py        #   GraphLLM_Connector (GNN↔LLM)
│   ├── train_proposed.py   #   Full model 학습
│   ├── train_llm_projector.py  # Projector 학습
│   ├── gardener_client_2.py    # GraphSAGE 인코더 학습
│   └── test_chat_llm.py       # LLM 추론 테스트
│
├── weights/                # 사전 학습 가중치
│   ├── gardener_encoder_latest.pth
│   ├── graph_projector_latest.pth
│   └── proposed_model_latest.pth
│
├── data/                   # RocksDB 저장소
├── docker-compose.yml
└── Dockerfile
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Database | RocksDB (Persistent) + LRU Cache (In-Memory) |
| Vector Search | FAISS (768-dim, Inner Product) |
| GNN | PyTorch Geometric (GCNConv, SAGEConv) |
| LLM | Qwen-2.5-1.5B + LoRA Fine-tuning |
| Embedding | jhgan/ko-sroberta-multitask |
| C++↔Python | pybind11 |
| API Server | FastAPI + Uvicorn |
| Container | Docker Compose |

## Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + CUDA (학습 시)

### Docker (권장)

```bash
docker-compose up --build
```

### Manual Build

```bash
# 시스템 의존성
apt install librocksdb-dev libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev

# C++ 확장 빌드
cd src_cpp && pip install .

# 서버 실행
cd src_server && uvicorn db_server_v2:app --host 0.0.0.0 --port 9000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upsert` | POST | 노드 추가 (중복 제거 포함) |
| `/search_vector` | POST | 벡터 유사도 검색 (top-k) |
| `/get_neighbors` | GET | 인접 노드 조회 |
| `/add_edge` | POST | 엣지 추가 |
| `/add_edge_batch` | POST | 배치 엣지 추가 |
| `/get_training_batch` | GET | 학습용 배치 데이터 |
| `/stats` | GET | DB 통계 |

## Training Pipeline

```
Phase 1: Gardener (GraphSAGE Encoder)
    → 그래프 구조 학습, 노드 임베딩 압축 (768 → 128)
    → gardener_client_2.py

Phase 2: LLM Projector
    → GNN 출력 → LLM 입력 공간 변환 학습
    → train_llm_projector.py

Phase 3: Full Model
    → GNN + Transformer Decoder 통합 학습
    → train_proposed.py
```

## Data Pipeline

```
Wikipedia XML Dump
    → wiki_dump_parser.py (문서 추출)
    → wiki_full_pipeline.py (임베딩 생성 + DB 적재)
    → 자동 엣지 생성 (cosine similarity ≥ 0.75)
    → FAISS 인덱스 구축
```

## Design Highlights

- **하이브리드 검색**: 벡터 유사도(FAISS) + 그래프 관계(GNN)를 결합한 검색
- **자동 엣지 생성**: 코사인 유사도 기반으로 유사 노드 간 엣지 자동 연결
- **2-Tier 캐싱**: LRU In-Memory Cache (2000 노드) + RocksDB On-Disk
- **커스텀 직렬화**: 파이프 구분자 기반 경량 포맷으로 RocksDB 저장
- **모듈형 학습**: 각 컴포넌트(인코더, 프로젝터, LLM)를 독립적으로 학습 가능

## API 동작 검증

서버의 주요 엔드포인트가 2.45M 노드 규모에서 정상적으로 응답하는 것을 확인했다.

<details>
<summary><b>검증 결과: test_api_demo.py 실행 로그</b></summary>

```
$ python3 test_api_demo.py

======================================================================
  API Endpoint Test
  Hybrid Graph-Vector DB Server (FastAPI)
======================================================================

[1/4] GET /node/{id} — RocksDB 직접 조회
--------------------------------------------------
  ID        0 → 768-dim | "[1번] 마라탕 맛이 궁금해......"
  ID   100000 → 768-dim | "[2번] 예전에 잃어버린 강아지 이야기 기억나요?..."
  ID  1000000 → 768-dim | "[Wiki: 시리아인] "레반트인들과 높은 유사성"을 나타냈다. ..."

[2/4] POST /search_vector — FAISS 벡터 검색
--------------------------------------------------
  Query: Node 0의 embedding으로 Top-3 검색
  #1 ID:  1186989 | Score: 1.0000 | "[Wiki: 키치너 (온타리오주)] ..."
  #2 ID:        0 | Score: 1.0000 | "[1번] 마라탕 맛이 궁금해......"
  #3 ID:  1018112 | Score: 0.8760 | "[Wiki: 오스트레일리아의 경제] ..."

[3/4] GET /get_neighbors — 그래프 엣지 조회
--------------------------------------------------
  Node 0의 이웃: 0개

[4/4] GET /get_training_batch — GNN 학습용 배치
--------------------------------------------------
  요청: batch_size=32
  응답: nodes=32, edges=16
  임베딩 차원: [768, 768, 768, 768, 768]

======================================================================
  모든 API 엔드포인트 정상 응답 확인
======================================================================
```

</details>

노드 조회, 벡터 검색, 그래프 탐색, 학습 배치 API가 모두 응답한다. 단, 벡터 검색 결과에서 Node 0("마라탕")을 검색했을 때 Score 1.0인 1위가 "키치너 (온타리오주)"로 반환되는 것은 아래 Post-Mortem에서 분석하는 인덱스 밀림 현상이다.

## Post-Mortem: Index Misalignment

2.45M 노드 규모에서 운영 중 **FAISS 인덱스와 RocksDB 간의 ID 매핑이 어긋나는 현상**이 발생하여 검색 정확도가 심각하게 저하되었다. 이 문제의 근본 원인은 커스텀 하이브리드 DB 설계에서 분산 시스템의 핵심 원칙들이 결여된 데 있었다.

<details>
<summary><b>검증 결과: test_index_mismatch.py 실행 로그 (5/5 MISMATCH)</b></summary>

```
$ python3 test_index_mismatch.py

================================================================================
  [Post-Mortem] Index Misalignment Test
  RocksDB 직접 조회 vs FAISS 벡터 검색 정합성 검증
================================================================================

[1/3] 서버 상태 확인
----------------------------------------
  서버 응답: OK

[2/3] 정합성 검증 (랜덤 노드 5개)
----------------------------------------

  ── Test 1 ── Node ID: 253645
     RocksDB  → ID:   253645  "[Wiki: 동우필름] 사)한국영화촬영감독(K.S.C.) 회원, ..."
     FAISS    → ID:  1228277  "[Wiki: 나사우 공국] 인리히 프리드리히 카를 폰 춤 슈타인의 ..."
     Score: 1.000000
     [MISMATCH] 요청 ID 253645 ≠ 반환 ID 1228277

  ── Test 2 ── Node ID: 1151796
     RocksDB  → ID:  1151796  "[Wiki: 오보카타 하루코] 철회하겠다고 발표했다. ..."
     FAISS    → ID:  2036612  "[Wiki: 의왕경찰서] 서장은 총경으로 보한다. ..."
     Score: 1.000000
     [MISMATCH] 요청 ID 1151796 ≠ 반환 ID 2036612

  ── Test 3 ── Node ID: 1693923
     RocksDB  → ID:  1693923  "[Wiki: 역사적 예수] 이유도 당시 지배 질서에 반대하는 ..."
     FAISS    → ID:   315884  "[Wiki: 중성미자] 조: 슈퍼 가미오칸데와 서드버리 ..."
     Score: 1.000000
     [MISMATCH] 요청 ID 1693923 ≠ 반환 ID 315884

  ── Test 4 ── Node ID: 2206639
     RocksDB  → ID:  2206639  "[Wiki: 박승철 (의사)] 복무하고 1973년 육군 소령으로 ..."
     FAISS    → ID:   777329  "[Wiki: 백관묵] 신의 임무로 여겨 세상에 문집 6권을 ..."
     Score: 1.000000
     [MISMATCH] 요청 ID 2206639 ≠ 반환 ID 777329

  ── Test 5 ── Node ID: 1848743
     RocksDB  → ID:  1848743  "[Wiki: 서로마 제국의 몰락] 다시 일어설 힘을 얻지 ..."
     FAISS    → ID:   455221  "[Wiki: 발다] 발다(Balda)는 독일 드레스덴의 사진기 ..."
     Score: 1.000000
     [MISMATCH] 요청 ID 1848743 ≠ 반환 ID 455221

[3/3] 최종 결과
================================================================================
  MISMATCH: 5 / 5

  CONCLUSION: 인덱스 밀림(Index Misalignment) 확인
  ─────────────────────────────────────────────
  RocksDB와 FAISS 간 ID 매핑이 어긋나 있음.
  유사도 1.0으로 정확히 검색되었으나, 반환된 노드의
  ID와 텍스트가 요청한 것과 일치하지 않음.

  ROOT CAUSE:
    - RocksDB(disk)와 FAISS(memory)의 Write Atomicity 부재
    - Bulk Insert 중 부분 실패로 인한 Ghost Data 발생
    - 서버 재시작 시 ID 시퀀스 불일치
================================================================================
```

</details>

서버 시작 로그에서도 불일치가 확인된다:
```
[Warning] Sync Mismatch! DB: 2454054 vs FAISS: 2454052
```

RocksDB에는 2,454,054개 노드가 존재하지만 FAISS에는 2,454,052개만 인덱싱되어 있어, 2개의 Ghost Data가 ID 매핑 전체를 오염시켰다.

### 1. Atomicity 부재 — 트랜잭션 없는 다중 저장소 쓰기

`/upsert` 한 번에 3개의 독립적인 쓰기가 순차 실행된다:

```
db.add_node(...)       # 1. RocksDB (disk, durable)
index.add(vec)         # 2. FAISS (memory, volatile)
ALL_VALID_IDS.append() # 3. Python list (memory, volatile)
```

이 세 연산이 하나의 트랜잭션으로 묶이지 않았기 때문에, Bulk Insert 도중 OOM이나 프로세스 크래시 발생 시 **RocksDB에는 기록되었지만 FAISS에는 반영되지 않은 "Ghost Data"** 가 생겼다. 재시작 후 ID 시퀀스가 한 칸씩 밀리면서 검색 결과가 엉뚱한 노드를 반환했다.

### 2. Concurrency 미흡 — Race Condition

```python
new_id = len(ALL_VALID_IDS)  # Thread-safe하지 않음
```

FastAPI의 비동기 환경에서 동시 요청이 들어올 경우, 두 스레드가 같은 `len` 값을 읽고 동일한 ID로 데이터를 쓰는 경쟁 상태가 발생했다. 이로 인해 FAISS 인덱스 개수와 실제 데이터 개수가 불일치했다.

### 3. Durability 격차 — FAISS의 휘발성

RocksDB는 매 쓰기마다 디스크에 즉시 기록(WAL)되지만, FAISS는 순수 인메모리로 동작하며 주기적인 `faiss.write_index` 호출에 의존했다. 서버가 비정상 종료되면 **마지막 저장 이후의 FAISS 변경분이 유실**되어 RocksDB와의 상태 불일치가 필연적으로 발생했다.

### Lessons Learned

| 원칙 | 교훈 |
|------|------|
| **Atomicity** | 다중 저장소에 걸친 쓰기는 반드시 트랜잭션 또는 2PC(Two-Phase Commit)로 묶어야 한다 |
| **Concurrency** | 공유 상태(ID 카운터)는 Lock 또는 Atomic Operation으로 보호해야 한다 |
| **Durability** | 인메모리 인덱스(FAISS)에도 WAL 패턴을 적용하거나, 저장소와 동기적으로 flush해야 한다 |
| **Recovery** | 시작 시 두 저장소 간 정합성을 검증하고 자동 복구하는 로직이 필수적이다 |
| **설계 판단** | 대규모 데이터에서 커스텀 DB를 직접 구현하는 것보다, 트랜잭션을 지원하는 기존 솔루션(Milvus, Weaviate 등)을 채택하는 것이 올바른 선택이었다 |

### 실패 구조 분석

아래는 실패의 원인이 된 현재 아키텍처이다. Python 애플리케이션이 두 개의 독립적인 저장소(RocksDB, FAISS)에 직접 쓰기를 수행하며, 이 사이에 트랜잭션 보장 메커니즘이 존재하지 않는다.

```
                    ┌─────────────────┐
                    │  Python Server  │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 │           │           │
                 ▼           ▼           ▼
          ┌──────────┐ ┌──────────┐ ┌────────────┐
          │ RocksDB  │ │  FAISS   │ │ Python List│
          │ (Disk)   │ │ (Memory) │ │ (Memory)   │
          └──────────┘ └──────────┘ └────────────┘
           ✅ 성공      ⚡ 실패 가능   ⚡ 실패 가능

          → 트랜잭션 없음: 부분 실패 시 정합성 깨짐
          → Lock 없음: 동시 쓰기 시 Race Condition
          → WAL 없음: FAISS 변경분 유실 가능
```

## Retrospective: What I Would Change

이 프로젝트를 다시 설계한다면, 아래와 같이 변경할 것이다.

### 1. 커스텀 DB → 트랜잭션 지원 벡터 DB로 교체

가장 근본적인 변경이다. RocksDB + FAISS를 직접 조합하는 대신, 벡터 검색과 메타데이터 저장을 하나의 트랜잭션으로 처리하는 전용 솔루션을 채택한다.

```
                    ┌─────────────────┐
                    │  Python Server  │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Qdrant /      │
                    │   Milvus        │
                    │                 │
                    │  ┌───────────┐  │
                    │  │  Vector   │  │
                    │  │  Index    │  │
                    │  ├───────────┤  │
                    │  │  Payload  │  │
                    │  │  Storage  │  │
                    │  ├───────────┤  │
                    │  │  Graph    │  │
                    │  │  Metadata │  │
                    │  └───────────┘  │
                    │                 │
                    │  ✅ 단일 트랜잭션  │
                    │  ✅ 내장 WAL     │
                    │  ✅ Thread-safe │
                    └─────────────────┘
```

- **Qdrant**: Payload에 neighbor list를 저장하면 그래프 구조를 표현할 수 있고, 벡터 검색과 필터링을 단일 API로 처리 가능
- **Milvus**: 대규모 분산 환경에 적합하며, Partition 기반으로 데이터 격리 가능

### 2. ID 관리 전략 변경

```python
# Before: 순차 정수 ID (취약)
new_id = len(ALL_VALID_IDS)

# After: UUID 기반 (충돌 불가능, 저장소 간 동기화 불필요)
import uuid
new_id = str(uuid.uuid4())
```

순차 정수 ID는 "FAISS 인덱스 순서 = DB ID 순서"라는 암묵적 가정에 의존했다. UUID를 사용하면 ID 자체가 고유하므로 저장소 간 순서 동기화가 불필요해진다.

### 3. GNN 파이프라인 개선

| 현재 | 개선안 |
|------|--------|
| graph_token 1개로 LLM에 컨텍스트 주입 | Multi-token projection (4~8 tokens)으로 정보량 확대 |
| 1-hop 이웃만 서브그래프로 사용 | 2-hop 확장 + Personalized PageRank로 중요 노드 선별 |
| Gardener가 엣지를 DB에 직접 전송 | 배치 단위로 검증 후 반영 (엣지 품질 필터링) |

### 4. 요약

```
현재:  Python ──→ [RocksDB] + [FAISS] + [List]  (3개 독립 저장소, 수동 동기화)
개선:  Python ──→ [Qdrant/Milvus]                (단일 저장소, 내장 트랜잭션)
```

핵심 교훈은 명확하다: **인프라 레이어의 정합성 문제를 애플리케이션 코드로 해결하려 하지 말 것.** 트랜잭션, 동시성 제어, 내구성은 검증된 저장소 엔진에 위임하고, 애플리케이션은 비즈니스 로직(GNN 학습, LLM 연동)에 집중해야 한다.