# Inference Engine

> GNN 기반 그래프 추론과 LLM 텍스트 생성을 결합한 추론 엔진

## Overview

사용자의 질문(query)을 받아 지식 그래프에서 관련 서브그래프를 추출하고, GNN으로 구조적 컨텍스트를 압축한 뒤, 이를 LLM에 직접 주입하여 답변을 생성하는 추론 파이프라인이다.

```
Query → Embedding → DB Search → Subgraph → GNN → Projector → LLM → Answer
```

## Model Architecture

### SharedGCNEncoder (공유 백본)

Gardener와 HybridQueryGNN이 동일한 가중치를 공유하는 GCN 인코더이다. 정보 손실을 방지하기 위해 히든 차원을 입력보다 크게 설정했다.

```
Input (768-dim)
  → GCNConv(768 → 1024) + ReLU
  → GCNConv(1024 → 768)
Output (768-dim)
```

### HybridQueryGNN (핵심 추론 모듈)

쿼리 벡터와 서브그래프를 동시에 받아, Local(노드 레벨)과 Global(그래프 레벨) 두 경로로 정보를 처리한다.

```
                    ┌─────────────────────────────────────────┐
                    │           HybridQueryGNN                │
                    │                                         │
[Node Embeddings]───┤  SharedGCNEncoder ──→ h_local (768)     │
                    │                          │              │
[Query Embedding]───┤  Linear(768→768) ──→ h_global (768)     │
                    │                          │              │
                    │         Scaled Dot-Product Attention     │
                    │         scores = (h_global · h_local)/√d│
                    │                          │              │
                    │         Weighted Sum ──→ message         │
                    │                          │              │
                    │         GRU Update ──→ h_global_new      │
                    │                          │              │
                    │         Broadcast + Residual             │
                    │              ──→ h_final (per node)      │
                    └─────────────────────────────────────────┘

Output:
  - h_final: 노드별 컨텍스트 임베딩 (768-dim × N nodes)
  - h_global: 그래프 전체 요약 벡터 (768-dim)
  - attention_weights: 노드별 중요도 가중치
```

**Local 경로**: SharedGCNEncoder가 이웃 노드의 정보를 집계하여 노드별 표현을 생성
**Global 경로**: 질문 벡터를 초기 상태로, Attention으로 중요 노드를 선별한 뒤 GRU로 글로벌 컨텍스트를 갱신

### Projector (GNN → LLM Bridge)

GNN의 768차원 출력을 LLM의 임베딩 공간(2560차원)으로 변환하는 2-layer MLP이다.

```
GNN h_global (768)
  → Linear(768 → 2560) + LayerNorm + GELU
  → Linear(2560 → 2560)
Output: graph_token (2560-dim, 1 token)
```

이 `graph_token`이 텍스트 임베딩 시퀀스의 맨 앞에 결합(concatenate)되어, LLM이 그래프 컨텍스트를 조건으로 답변을 생성한다.

```
LLM Input = [graph_token] + [text_input_embeddings]
             └── GNN이 압축한 구조적 지식
```

### GraphLLM_Connector (Projector 학습 모델)

GNN(frozen) + Projector(trainable) + LLM(LoRA) 세 모듈을 연결하는 학습 프레임워크이다.

```
┌──────────────────────────────────────────────────┐
│              GraphLLM_Connector                   │
│                                                   │
│  HybridQueryGNN (frozen) ──→ h_global (768)       │
│                                 │                 │
│  Projector (trainable)          │                 │
│    Linear(768→2560) + LN + GELU │                 │
│    Linear(2560→2560)            │                 │
│                                 ▼                 │
│              graph_token (1 × 2560)               │
│                     +                             │
│              text_embeddings (N × 2560)           │
│                     │                             │
│                     ▼                             │
│           Qwen-2.5-1.5B (LoRA)                    │
│           rank=8, alpha=32                        │
│           target: q_proj, v_proj                  │
│                     │                             │
│                     ▼                             │
│              Answer Tokens                        │
└──────────────────────────────────────────────────┘
```

### GraphRAG_Generator (독립형 생성 모델)

LLM 없이 GNN + Transformer Decoder로 직접 텍스트를 생성하는 모델이다.

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

- 토크나이저: `bert-base-multilingual-cased` (vocab 30,522)
- 200스텝마다 Gardener의 최신 가중치를 동기화(Brain Sync)

## Training Pipeline

### Phase 1: Gardener (GraphSAGE Encoder)

그래프 구조를 학습하여 노드 임베딩을 128차원으로 압축한다.

→ 상세 내용은 [Gardener 문서](gardener.md) 참조

### Phase 2: LLM Projector

GNN 출력을 LLM 입력 공간으로 변환하는 Projector를 학습한다.

```
학습 데이터: train_snapshot_clean.pt
  - 노드 임베딩 (768-dim)
  - self-loop 엣지 (단일 노드 그래프)
  - 쿼리 임베딩 (노이즈 추가된 원본)
  - 타겟 텍스트 (노드의 원본 텍스트)
```

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Gradient Accumulation | 4 steps |
| Learning Rate | 2e-4 |
| Optimizer | AdamW |
| Max Steps | 2,000 |
| LLM | Qwen-2.5-1.5B-Instruct (4-bit) |
| LoRA | rank=8, alpha=32, target=q_proj/v_proj |
| GNN | Frozen |
| Projector | Trainable |

**학습 흐름:**
1. 배치에서 노드 임베딩 + 쿼리 추출
2. GNN 인코딩 → `h_global` (768-dim)
3. Projector → `graph_token` (2560-dim)
4. `[graph_token] + [text_embeddings]` → LLM
5. CrossEntropy Loss (graph_token 위치는 무시)
6. Projector + LoRA만 역전파

### Phase 3: Full Model (GraphRAG_Generator)

DB 서버에서 실시간으로 배치를 가져와 GNN + Decoder를 통합 학습한다.

| Parameter | Value |
|-----------|-------|
| Batch Size | 4,096 (HTTP fetch) |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Total Steps | 100K+ |
| Decoder | 3-layer Transformer (4 heads) |
| Brain Sync | 매 200스텝 Gardener 가중치 로드 |

**학습 흐름:**
1. `/get_training_batch`로 서브그래프 배치 수신
2. 유효 노드 필터링 (텍스트 길이 > 5, 임베딩 768차원)
3. 타겟 노드 선택 + 노이즈 쿼리 생성
4. GNN → Memory → Transformer Decoder
5. Teacher Forcing으로 다음 토큰 예측

## Inference Flow

### LLM 기반 추론

```python
# 1. 쿼리 임베딩 생성
query_emb = embedding_model.encode(query_text)  # 768-dim

# 2. DB에서 유사 노드 검색
results = db.search_vector(query_emb, k=5)

# 3. 서브그래프 구성
x = torch.tensor([node.embedding for node in results])      # [5, 768]
edge_index = build_edges(results)                            # [2, E]

# 4. GNN 인코딩
h_nodes, h_global, attn = gnn(x, edge_index, query_emb)     # h_global: [768]

# 5. LLM 공간으로 투영
graph_token = projector(h_global).unsqueeze(0)               # [1, 1, 2560]

# 6. 텍스트 임베딩과 결합
prompt = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
input_embeds = llm.get_input_embeddings()(tokenize(prompt))  # [1, N, 2560]
combined = torch.cat([graph_token, input_embeds], dim=1)     # [1, N+1, 2560]

# 7. LLM 생성
output = llm(inputs_embeds=combined)
```

### 독립형 추론 (GraphRAG_Generator)

```python
# 1~4: 동일
# 5. Transformer Decoder로 직접 생성
memory = torch.cat([h_global, h_nodes], dim=0)  # [1+N, 768]
tokens = model.generate(memory, max_length=64)   # Autoregressive
text = tokenizer.decode(tokens)
```

## Memory Profile

| Component | Memory |
|-----------|--------|
| Qwen-2.5-1.5B (4-bit) | ~2 GB |
| GNN (HybridQueryGNN) | < 50 MB |
| Projector | < 10 MB |
| FAISS Index (2.4M vectors) | ~7 GB |

## Weights

학습된 가중치 파일이다 (`.gitignore`로 제외됨).

| File | Description |
|------|-------------|
| `gardener_encoder_latest.pth` | GraphSAGE 인코더 (128-dim 출력) |
| `graph_projector_latest.pth` | Projector + LoRA 어댑터 |
| `proposed_model_latest.pth` | GraphRAG_Generator 전체 |

## File Map

```
src_python/
├── models.py               # SharedGCNEncoder, HybridQueryGNN 정의
├── model_llm.py            # GraphLLM_Connector (GNN↔LLM 브릿지)
├── train_llm_projector.py  # Phase 2: Projector + LoRA 학습
├── train_proposed.py       # Phase 3: Full Model 통합 학습
├── my_hybrid_client.py     # DB 서버 HTTP 클라이언트
├── refresh_vectors.py      # 임베딩 재생성 + 스냅샷 생성
├── test_chat_llm.py        # LLM 추론 테스트
├── test_chat.py            # 채팅 테스트
├── test_db.py              # DB 연결 테스트
└── Dockerfile              # PyTorch + CUDA 런타임
```
