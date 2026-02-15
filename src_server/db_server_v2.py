from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import faiss
import numpy as np
import os
import logging
import random
import gc
import ctypes
import sys
import orjson
from apscheduler.schedulers.background import BackgroundScheduler

# ==========================================================
# ⚙️ 환경 설정 및 라이브러리 로드
# ==========================================================
sys.path.append("/app/build")
sys.path.append("/workspace/build")

try:
    import my_hybrid_backend as mydb
except ImportError:
    try:
        import graph_db as mydb
    except ImportError:
        print("❌ [Error] DB Backend Module not found!")
        sys.exit(1)

from metadata_manager import load_node_count, save_node_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BrainServer] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("BrainServer")

app = FastAPI()

DB_PATH = "data/my_conversation_db"
INDEX_PATH = "data/my_index.faiss"

# 1. RocksDB 로드
if not os.path.exists("data"): os.makedirs("data")
logger.info(f"📂 Opening RocksDB at {DB_PATH}...")

try:
    # Threshold와 Capacity는 필요에 따라 조정
    db = mydb.HybridGraphDB(path=DB_PATH, threshold=0.75, capacity=20000)
except Exception as e:
    logger.error(f"❌ Failed to load DB backend: {e}")
    sys.exit(1)

# 2. FAISS 로드 (CPU 모드)
logger.info(f"🧠 Loading FAISS Index...")
if os.path.exists(INDEX_PATH):
    try: 
        index = faiss.read_index(INDEX_PATH)
        logger.info(f"   ✅ Loaded existing index with {index.ntotal} vectors.")
    except: 
        logger.warning("   ⚠️ Failed to read index, creating new one.")
        index = faiss.IndexFlatIP(768)
else:
    logger.info("   🆕 Creating new FAISS Index...")
    index = faiss.IndexFlatIP(768)

ALL_VALID_IDS = []
scheduler = BackgroundScheduler()

@app.on_event("startup")
def startup_event():
    global ALL_VALID_IDS
    if not scheduler.running:
        scheduler.start()
        logger.info("[System] Scheduler started.")

    try:
        logger.info("[Startup] Loading all node IDs map... (This may take a moment)")
        # DB에 있는 모든 ID를 가져와서 리스트로 관리 (FAISS Index 순서와 일치해야 함)
        ALL_VALID_IDS = db.get_all_node_ids()
        
        # FAISS 인덱스와 DB 개수 불일치 시 경고
        if len(ALL_VALID_IDS) != index.ntotal:
             logger.warning(f"[Warning] Sync Mismatch! DB: {len(ALL_VALID_IDS)} vs FAISS: {index.ntotal}")
        
        count = len(ALL_VALID_IDS)
        save_node_count(count)
        logger.info(f"[Startup] ✅ Ready! Total nodes: {count}")
    except Exception as e:
        logger.error(f"[Startup] Failed to load node IDs: {e}")
        ALL_VALID_IDS = []

# ==========================================================
# 📦 데이터 모델 (Pydantic)
# ==========================================================
class NodeInput(BaseModel):
    text: str
    embedding: list[float]

class UpsertInput(BaseModel):
    text: str
    embedding: list[float]
    threshold: float = 0.92 

class SearchInput(BaseModel):
    vector: list[float]
    k: int = 5
    
class EdgeRequest(BaseModel):
    src: int
    dst: int
    weight: float

class EdgeBatchItem(BaseModel):
    src: int
    dst: int
    weight: float
    type: str = "knn"

# ==========================================================
# 🚀 API 정의
# ==========================================================

# ★ [핵심 기능 1] 서버 사이드 벡터 검색 (Server-Side Retrieval)
# 클라이언트가 쿼리 벡터만 보내면, 서버가 검색 후 결과(Top-K)만 리턴
@app.post("/search_vector")
def search_vector_api(item: SearchInput):
    global ALL_VALID_IDS
    try:
        # 1. 쿼리 벡터 준비 (정규화)
        q = np.array([item.vector], dtype=np.float32)
        faiss.normalize_L2(q)
        
        # 2. FAISS 검색
        D, I = index.search(q, item.k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            
            # FAISS Index ID -> Real DB ID 매핑
            # (현재 구조상 순차적으로 추가되므로 인덱스가 곧 ID 목록의 인덱스라고 가정)
            if idx < len(ALL_VALID_IDS):
                real_id = ALL_VALID_IDS[idx]
                score = float(D[0][i])
                
                # DB에서 상세 정보 조회
                try:
                    node = db.get_node(real_id)
                    results.append({
                        "id": real_id,
                        "text": node.text,
                        "embedding": node.embedding, # GNN 입력을 위해 임베딩도 반환
                        "score": score
                    })
                except: pass
                
        return {"status": "ok", "results": results}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"status": "error", "msg": str(e)}

# ★ [핵심 기능 2] 이웃 노드 조회 (GNN 확장용)
@app.get("/get_neighbors")
def get_neighbors_api(node_id: int):
    try:
        # [(neighbor_id, weight), ...] 형태 반환
        neighbors = db.get_neighbors(node_id)
        return neighbors
    except Exception as e:
        return []

# Upsert: 중복 방지 지식 추가
@app.post("/upsert")
def upsert_node(item: UpsertInput):
    global ALL_VALID_IDS
    try:
        # 1. 유사도 검색으로 중복 체크
        vec = np.array([item.embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        
        if index.ntotal > 0:
            D, I = index.search(vec, 1)
            best_score = float(D[0][0])
            best_idx = int(I[0][0])
            
            # 이미 아는 내용이면 (Threshold 초과) 스킵하고 기존 ID 반환
            if best_score > item.threshold and best_idx != -1 and best_idx < len(ALL_VALID_IDS):
                existing_id = ALL_VALID_IDS[best_idx]
                return {
                    "status": "exists", 
                    "id": existing_id, 
                    "score": best_score, 
                    "msg": "Knowledge already exists."
                }
        else:
            best_score = 0.0

        # 2. 없으면 신규 추가
        new_id = len(ALL_VALID_IDS)
        if index.ntotal > 0:
             # ID 체계 무결성 확인 (혹시 모를 상황 대비)
             # 실제로는 ALL_VALID_IDS 길이와 index.ntotal은 같아야 함
             pass

        db.add_node(new_id, item.text, item.embedding)
        
        index.add(vec)
        ALL_VALID_IDS.append(new_id)
        save_node_count(len(ALL_VALID_IDS))
        
        # 주기적으로 인덱스 디스크 저장 (선택 사항)
        if new_id % 1000 == 0:
            faiss.write_index(index, INDEX_PATH)
        
        return {
            "status": "created", 
            "new_id": new_id,
            "node_id": new_id, # 클라이언트 호환성 위해 둘 다 제공
            "score": best_score,
            "msg": "New knowledge added to Graph."
        }
        
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 대용량 임베딩 덤프 (Cache Builder용)
@app.get("/get_embeddings_bulk")
def get_embeddings_bulk(limit: int = 20000):
    global ALL_VALID_IDS
    if not ALL_VALID_IDS:
        try: ALL_VALID_IDS = db.get_all_node_ids()
        except: return Response(content=b'{"nodes": []}', media_type="application/json")

    # limit만큼 랜덤 샘플링 또는 전체 반환
    if limit >= len(ALL_VALID_IDS):
        target_ids = ALL_VALID_IDS
    else:
        # 학습용 데이터라면 랜덤이 좋음
        target_ids = random.sample(ALL_VALID_IDS, limit)
    
    result_nodes = []
    for nid in target_ids:
        try:
            node = db.get_node(nid)
            if node.id != -1:
                result_nodes.append({
                    "id": node.id, 
                    "embedding": node.embedding 
                })
        except: pass
    
    json_bytes = orjson.dumps({"nodes": result_nodes})
    return Response(content=json_bytes, media_type="application/json")

@app.post("/add_edges_batch")
def add_edges_batch(items: list[EdgeBatchItem]):
    try:
        count = 0
        for item in items:
            db.add_edge(item.src, item.dst, item.weight)
            count += 1
        return {"status": "success", "added": count}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.get("/node/{node_id}")
def get_node(node_id: int):
    try:
        node = db.get_node(node_id)
        if node.id == -1: raise HTTPException(status_code=404)
        return {"id": node.id, "text": node.text, "embedding": node.embedding}
    except: raise HTTPException(status_code=404)

@app.get("/get_node")
def get_node_query(node_id: int): # 쿼리 파라미터 버전 (?node_id=123)
    return get_node(node_id)

@app.post("/add_edge")
def add_edge_api(req: EdgeRequest):
    try:
        db.add_edge(req.src, req.dst, req.weight)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Add edge failed: {e}")
        return {"status": "error"}

@app.get("/get_training_batch")
def get_training_batch(batch_size: int = 256):
    global ALL_VALID_IDS
    if not ALL_VALID_IDS: return {"nodes": [], "edges": []}

    seed_size = batch_size // 2
    seed_ids = random.sample(ALL_VALID_IDS, min(seed_size, len(ALL_VALID_IDS)))
    
    selected_ids = set(seed_ids)
    edges_list = []
    
    for src in seed_ids:
        try:
            neighbors = db.get_neighbors(src)
            for dst, w in neighbors:
                edges_list.append({"src": src, "dst": dst, "weight": w})
                if len(selected_ids) < batch_size:
                    selected_ids.add(dst)
        except: pass
    
    if len(selected_ids) < batch_size:
        remaining = batch_size - len(selected_ids)
        others = random.sample(ALL_VALID_IDS, min(remaining, len(ALL_VALID_IDS)))
        selected_ids.update(others)

    nodes_list = []
    final_id_set = set()
    
    for nid in selected_ids:
        try:
            node = db.get_node(nid)
            if node.id != -1:
                final_id_set.add(node.id)
                nodes_list.append({
                    "id": node.id,
                    "embedding": node.embedding,
                })
        except: pass

    final_edges = []
    for e in edges_list:
        if e['src'] in final_id_set and e['dst'] in final_id_set:
            final_edges.append(e)

    return {"nodes": nodes_list, "edges": final_edges}

@app.post("/cleanup_memory")
def cleanup_memory():
    try:
        db.clear_cache()
        gc.collect()
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)
        except: pass
        return {"status": "Cleaned"}
    except Exception as e: return {"error": str(e)}

def gardener_task(): 
    # 주기적으로 인덱스 저장
    if index.ntotal > 0:
        faiss.write_index(index, INDEX_PATH)

scheduler.add_job(gardener_task, 'interval', minutes=20)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)