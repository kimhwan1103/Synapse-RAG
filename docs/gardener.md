# Gardener: GNN-based Graph Retriever

> 정원사처럼 지식 그래프의 연결을 가꾸는 GraphSAGE 기반 링크 예측 시스템

## Overview

Gardener는 그래프의 위상학적 구조를 학습하여 새로운 의미적 연결을 발견하고, 기존 연결을 강화하는 자율 에이전트이다. GAE(Graph Auto-Encoder) 프레임워크에서 GraphSAGE 인코더를 학습하고, 학습된 잠재 공간에서 FAISS 검색으로 새로운 엣지를 탐색한다.

```
┌─────────────────────────────────────────────────────────────┐
│                      Gardener Loop                          │
│                                                             │
│  1. DB에서 서브그래프 배치 수신 (2048 nodes)                     │
│  2. GraphSAGE로 128차원 잠재 공간으로 인코딩                     │
│  3. GAE Reconstruction Loss로 학습                           │
│  4. FAISS로 잠재 공간에서 유사 노드 탐색                         │
│  5. 발견된 엣지를 DB에 전송                                    │
│  6. → 1로 반복                                              │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### GraphSAGE Encoder

768차원 임베딩을 128차원 잠재 공간으로 압축한다. 이 잠재 공간에서 원본 벡터로는 발견하지 못한 의미적 연결을 탐색한다.

```
Input (768-dim)
  → Linear(768 → 256) + BatchNorm + ELU + Dropout(0.1)
  → SAGEConv(256 → 256) + BatchNorm + ELU + Dropout(0.1)
  → SAGEConv(256 → 128)
  → L2 Normalize × 3.0 (Max Norm Scaling)
Output (128-dim)
```

**설계 의도:**
- **차원 축소 (768 → 128)**: 83% 압축. 고차원에서 가려진 의미적 관계를 저차원에서 드러냄
- **BatchNorm**: 학습 안정성
- **ELU**: 음수 영역에서도 기울기를 유지하여 dead neuron 방지
- **Max Norm Scaling (×3.0)**: 임베딩 크기를 제한하여 FAISS 검색의 안정성 확보

### GAE (Graph Auto-Encoder)

인코더의 출력으로 엣지 존재 여부를 예측하는 링크 예측 모델이다.

```
Training:
  z = Encoder(x, edge_index)     # 노드 임베딩 → 128-dim
  loss = GAE.recon_loss(z, pos_edge_index) + GAE.kl_loss()

Inference:
  z = Encoder(x, edge_index)     # 학습된 잠재 공간
  → FAISS 검색으로 유사 노드 발견 → 새 엣지 생성
```

## 2-Phase Link Discovery

### Phase 1: Raw Embedding Linking

원본 768차원 임베딩으로 직접 FAISS 검색을 수행한다.

```
Node Embeddings (768-dim)
    │
    ▼ L2 Normalize
    │
    ▼ FAISS IndexFlatIP (Inner Product)
    │  K=5 nearest neighbors
    │
    ▼ Dynamic Threshold: avg + 2σ (범위: 0.75 ~ 0.96)
    │
    ▼ Edge Type: "knn_global"
```

### Phase 2: Semantic Deep Linking

학습된 GraphSAGE 인코더로 128차원으로 압축한 뒤 검색한다. Phase 1이 놓친 구조적 연결을 발견한다.

```
Node Embeddings (768-dim)
    │
    ▼ GraphSAGE Encoder
    │
    ▼ Learned Embeddings (128-dim)
    │
    ▼ L2 Normalize
    │
    ▼ FAISS IndexFlatIP (Inner Product)
    │  K=5 nearest neighbors
    │
    ▼ Dynamic Threshold: avg + 2σ (범위: 0.85 ~ 0.98)
    │
    ▼ Edge Type: "knn_semantic"
```

### Phase 비교

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| 임베딩 차원 | 768 (원본) | 128 (학습) |
| 유사도 임계값 | 0.75 ~ 0.96 | 0.85 ~ 0.98 |
| 발견하는 관계 | 표면적 유사성 | 구조적/의미적 관계 |
| 엣지 타입 | `knn_global` | `knn_semantic` |
| 계산 비용 | 낮음 | 높음 (인코더 추론 필요) |

## Dynamic Threshold

고정 임계값 대신, 배치별 유사도 분포에 적응하는 동적 임계값을 사용한다.

```python
scores = faiss_results.distances  # 배치의 모든 유사도 점수

avg_score = scores.mean()
std_score = scores.std()

dynamic_threshold = min(max_val, max(min_val, avg_score + 2 * std_score))
```

- 유사도가 전반적으로 높은 영역 → 임계값 상승 → 정밀도 유지
- 유사도가 낮은 영역 → 임계값 하강 → 재현율 유지

## Edge Lifecycle

```
Discovery ──→ Deduplication ──→ Batch Transmission ──→ Persistence
                                                          │
                                                    Training에서
                                                    weight > 0.1만
                                                    사용 (암묵적 소멸)
```

### 1. Discovery (발견)

FAISS K-NN 검색으로 후보 엣지를 생성한다.

### 2. Deduplication (중복 제거)

```python
# 양방향 키로 정규화
edge_key = tuple(sorted((src_id, dst_id)))

# 인메모리 캐시로 중복 검사 (O(1))
if edge_key in sent_edges_cache:
    skip  # 이미 전송한 엣지

# 캐시 크기 제한 (1M 초과 시 초기화)
if len(sent_edges_cache) > 1_000_000:
    sent_edges_cache.clear()
```

### 3. Batch Transmission (전송)

```python
payload = [
    {"src": 100, "dst": 200, "weight": 0.89, "type": "knn_semantic"},
    {"src": 100, "dst": 300, "weight": 0.85, "type": "knn_semantic"},
    ...
]

# 우선: 배치 전송 (timeout: 2.0s)
POST /add_edges_batch  →  payload

# 폴백: 개별 전송 (timeout: 0.1s each)
POST /add_edge  →  each edge
```

### 4. Implicit Decay (암묵적 소멸)

명시적인 엣지 삭제/감쇠 메커니즘은 없다. 대신:
- 학습 시 `weight > 0.1` 이하 엣지는 자동 필터링
- 새로운 FAISS 검색에서 재발견되지 않는 엣지는 자연스럽게 비활성화
- 지속적인 재학습으로 의미적으로 유효한 엣지만 강화됨

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=300) |
| Min LR | 1e-6 |
| Batch Size | 2,048 nodes |
| Loss | GAE Reconstruction Loss |
| FAISS Linking | 매 10 스텝 |
| Weight Save | 매 200 스텝 |
| Index Update | 매 500 스텝 (background thread) |

### Training Loop

```
for step in range(∞):
    # 1. 배치 수신
    batch = GET /get_training_batch?batch_size=2048

    # 2. 전처리
    x = normalize(batch.embeddings)      # [2048, 768]
    edge_index = batch.edges             # [2, E]

    # 3. GAE 학습
    z = encoder(x, edge_index)           # [2048, 128]
    loss = gae.recon_loss(z, edge_index)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # 4. FAISS 링킹 (10스텝마다)
    if step % 10 == 0:
        perform_faiss_linking(z)  # Phase 2
        perform_faiss_linking(x)  # Phase 1

    # 5. 가중치 저장 (200스텝마다)
    if step % 200 == 0:
        save("gardener_encoder_latest.pth")

    # 6. FAISS 인덱스 갱신 (500스텝마다, 백그라운드)
    if step % 500 == 0:
        Thread(update_faiss_index).start()
```

### Brain Sync

GraphRAG_Generator가 매 200스텝마다 Gardener의 최신 가중치를 로드하여 GNN 인코더를 동기화한다.

```
Gardener (학습 중)
    │  gardener_encoder_latest.pth
    │  (매 200스텝 저장)
    │
    ▼
GraphRAG_Generator
    │  (매 200스텝 로드)
    │  SharedGCNEncoder 가중치 동기화
    ▼
추론 시 최신 그래프 구조 반영
```

## Monitoring

### Real-Time Monitor (`monitor_edges.py`)

그래프에 새로 추가되는 엣지를 실시간으로 관찰한다.

```bash
python monitor_edges.py
```

```
# 출력 예시
[새 연결] 위키: 테슬라 모터스 ↔ 위키: 일론 머스크
[새 연결] 위키: 양자역학 ↔ 위키: 슈뢰딩거 방정식
.  ← 서버 응답 없음 (대기 중)
[새 연결] 위키: 서울특별시 ↔ 위키: 대한민국의 수도
```

- 1,000개 노드씩 샘플링
- 새로 발견된 엣지만 출력
- 메모리 관리: `seen_edges` 10,000개 초과 시 초기화

### Health Check (`check_edge.py`)

그래프에 엣지가 존재하는지 빠르게 확인한다.

```bash
python check_edge.py
```

```
# 출력 예시
노드 수: 2454054, 엣지 수: 16
첫 5개 연결:
  1001 → 1002 (weight: 0.87)
  1001 → 1003 (weight: 0.82)
  ...
```

## FAISS Index Management

### Background Update

메인 학습 루프를 차단하지 않도록 별도 스레드에서 FAISS 인덱스를 갱신한다.

```python
def update_faiss_index():
    # 1. DB에서 최대 20,000개 노드 임베딩 수신
    embeddings = GET /get_embeddings_bulk?limit=20000

    # 2. L2 정규화
    faiss.normalize_L2(embeddings)

    # 3. 새 인덱스 구축 (GPU 가속, 256MB temp memory)
    index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embeddings)

    # 4. 참조 업데이트
    global_index = gpu_index
```

### GPU Acceleration

```python
res = faiss.StandardGpuResources()
res.setTempMemory(256 * 1024 * 1024)  # 256MB GPU temp memory
```

## Error Handling & Resilience

| 상황 | 대응 |
|------|------|
| 네트워크 타임아웃 | 5초 대기 후 재시도 |
| 배치 전송 실패 | 개별 전송으로 폴백 |
| 잘못된 임베딩 차원 | 해당 노드 스킵 |
| 엣지 없는 배치 | 빈 edge_index로 안전 처리 (Cold Start) |
| GPU 미사용 | CPU 자동 폴백 |
| FAISS 업데이트 중 | 검색 일시 중단 (플래그 기반) |
| Loss < 1e-8 | 역전파 스킵 (수치 안정성) |

## File Map

```
src_python/
├── gardener_client.py      # Phase 1: 초기 링크 발견 (768-dim)
├── gardener_client_2.py    # Phase 2: 시맨틱 딥 링킹 (128-dim)
├── monitor_edges.py        # 실시간 엣지 모니터링
├── check_edge.py           # 엣지 존재 확인 (Health Check)
├── metadata_manager.py     # 메타데이터 관리 (node_count)
└── models.py               # SharedGCNEncoder 정의 (공유)
```
