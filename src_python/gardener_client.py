import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, SAGEConv
from torch_geometric.data import Data
import requests
import numpy as np
import logging
import time
import os
import faiss
import threading
import traceback

# ==============================================================================
# ⚙️ 설정
# ==============================================================================
WEIGHT_PATH = "/workspace/weights"
os.makedirs(WEIGHT_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Gardener] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Gardener")

# ==============================================================================
# 🔥 [CORE] GraphSAGE Encoder (Scaled Output)
# ==============================================================================
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. Projection + BatchNorm
        self.linear = nn.Linear(in_channels, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # 2. SAGE Layer 1 + BatchNorm
        self.conv1 = SAGEConv(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # 3. SAGE Layer 2 (Final)
        self.conv2 = SAGEConv(256, out_channels)

    def forward(self, x, edge_index):
        # Step 1
        x = self.linear(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Step 2
        x = self.conv1(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Step 3
        x = self.conv2(x, edge_index)
        
        # ★ [핵심] Normalize 후 3배 스케일링 (Max Norm = 3.0)
        return F.normalize(x, p=2, dim=1) * 3.0

# ==============================================================================
# 🌿 Gardener Client
# ==============================================================================
class RemoteGardener:
    def __init__(self, server_url="http://rag-db-server:9000"): 
        self.server_url = server_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화 (Link Prediction용 128차원 출력)
        self.model = GAE(GraphSAGEEncoder(768, 128)).to(self.device)
        self.load_weights()
        
        # 옵티마이저 & 스케줄러 (재시작 시 LR 0.001로 리셋됨)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6
        )
        
        # FAISS 관련 변수
        self.reference_nodes = []
        self.faiss_index = None
        self.is_updating_index = False 
        
        # ★ [추가] 중복 전송 방지용 캐시 (메모리 절약형 Set)
        self.sent_edges_cache = set()
        
        logger.info(f"🌿 Gardener initialized on {self.device} (Smart Deduplication Mode)")

    def load_weights(self):
        weight_file = os.path.join(WEIGHT_PATH, "gardener_encoder_latest.pth")
        if os.path.exists(weight_file):
            try:
                state_dict = torch.load(weight_file, map_location=self.device)
                self.model.encoder.load_state_dict(state_dict)
                logger.info("📂 Loaded existing Gardener weights.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load weights: {e}")

    def fetch_batch_data(self, batch_size=2048):
        try:
            res = requests.get(f"{self.server_url}/get_training_batch", params={"batch_size": batch_size}, timeout=60)
            if not res.ok: return None, None
            
            data_json = res.json()
            raw_nodes = data_json.get("nodes", [])
            raw_edges = data_json.get("edges", [])
            
            valid_nodes = [n for n in raw_nodes if len(n.get("embedding", [])) == 768]
            if len(valid_nodes) < 50: return None, None

            idx_to_id = {i: n["id"] for i, n in enumerate(valid_nodes)}
            id_to_idx = {n["id"]: i for i, n in enumerate(valid_nodes)}
            
            x = torch.tensor([n["embedding"] for n in valid_nodes], dtype=torch.float).to(self.device)
            x = F.normalize(x, p=2, dim=1) 
            
            src_list, dst_list = [], []
            for e in raw_edges:
                if e["src"] in id_to_idx and e["dst"] in id_to_idx:
                    # 엣지 필터링 (0.1 이상)
                    if e.get("weight", 1.0) > 0.1: 
                        u, v = id_to_idx[e["src"]], id_to_idx[e["dst"]]
                        src_list.append(u)
                        dst_list.append(v)
                        # 양방향 학습 (Undirected)
                        src_list.append(v)
                        dst_list.append(u)
            
            # Cold Start 지원: 엣지가 없어도 빈 텐서 반환 (FAISS 검색 수행 위해)
            if not src_list:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device)

            return Data(x=x, edge_index=edge_index), idx_to_id

        except Exception as e:
            logger.error(f"❌ Fetch Error: {e}")
            return None, None

    def update_global_faiss_index(self, max_nodes=20000):
        if self.is_updating_index: return
        self.is_updating_index = True
        
        try:
            # 타임아웃 넉넉히 설정
            res = requests.get(f"{self.server_url}/get_embeddings_bulk", params={"limit": max_nodes}, timeout=180)
            if not res.ok: 
                self.is_updating_index = False
                return

            data = res.json()
            nodes = data.get("nodes", [])
            
            # 불량 데이터 필터링
            valid_nodes = [n for n in nodes if isinstance(n.get("embedding"), list) and len(n["embedding"]) == 768]
            
            if not valid_nodes:
                self.is_updating_index = False
                return

            embeddings = np.array([n["embedding"] for n in valid_nodes], dtype=np.float32)
            
            # 인덱스 생성 (768차원)
            index = faiss.IndexFlatIP(768)
            if torch.cuda.is_available():
                res_gpu = faiss.StandardGpuResources()
                res_gpu.setTempMemory(256 * 1024 * 1024) 
                index = faiss.index_cpu_to_gpu(res_gpu, 0, index)
            
            index.add(embeddings)
            
            self.faiss_index = index
            self.reference_nodes = [n["id"] for n in valid_nodes]
            
            logger.info(f"✅ [Background] FAISS Index Refreshed. Valid Nodes: {len(valid_nodes)}")

        except Exception as e:
            logger.error(f"⚠️ FAISS Update Failed: {e}")
        finally:
            self.is_updating_index = False

    def run_continuous_training(self, steps=1000000):
        self.model.train()
        logger.info("🚀 Gardener started working...")
        
        self.update_global_faiss_index()

        for step in range(1, steps + 1):
            data, idx_to_id = self.fetch_batch_data(batch_size=2048)
            
            if not data:
                time.sleep(5)
                continue

            try:
                self.optimizer.zero_grad()
                z = self.model.encode(data.x, data.edge_index)
                
                # 엣지가 있을 때만 Loss 계산 및 역전파
                has_edges = data.edge_index.size(1) > 0
                loss_val = 0.0

                if has_edges:
                    loss = self.model.recon_loss(z, data.edge_index)
                    if loss.item() > 1e-8:
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step(loss)
                        loss_val = loss.item()
                
                if step % 20 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"   [Step {step}] Loss: {loss_val:.4f} | LR: {current_lr:.6f} | Edges: {data.edge_index.size(1)}")
                
                if step % 200 == 0:
                    save_path = os.path.join(WEIGHT_PATH, "gardener_encoder_latest.pth")
                    tmp_path = save_path + ".tmp"
                    torch.save(self.model.encoder.state_dict(), tmp_path)
                    os.replace(tmp_path, save_path)
                
                if step % 500 == 0:
                    threading.Thread(target=self.update_global_faiss_index, daemon=True).start()

                # ★ 중요: FAISS 검색에는 학습된 z(128차원)가 아니라 원본 x(768차원)를 사용
                if step % 10 == 0 and self.faiss_index is not None and not self.is_updating_index:
                    self.perform_faiss_linking(data.x, idx_to_id)
            
            except Exception as e:
                logger.error(f"Step failed: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)
    
    def perform_faiss_linking(self, x, idx_to_id): # 인자 x(768차원)
        with torch.no_grad():
            queries = x.detach().cpu().numpy().astype(np.float32)
            
            # 코사인 유사도를 위한 정규화
            faiss.normalize_L2(queries)
            
            k = 5 
            D, I = self.faiss_index.search(queries, k)
            
            avg_score = np.mean(D)
            std_score = np.std(D)
            
            # 동적 임계값 (최소 0.75, 최대 0.96)
            dynamic_threshold = min(0.96, max(0.75, avg_score + 2 * std_score))
            
            # Cold Start용 최소값 체크
            if np.max(D) < 0.7: return

            edges_to_send = []
            new_edges_count = 0
            
            for i in range(len(queries)):
                src_id = idx_to_id.get(i)
                if not src_id: continue
                
                for j in range(k):
                    score = float(D[i, j])
                    if score < dynamic_threshold: continue
                    
                    target_idx = I[i, j]
                    if target_idx == -1: continue
                    
                    dst_id = self.reference_nodes[target_idx]
                    if src_id == dst_id: continue

                    # ★ [핵심] 중복 엣지 필터링 (방향 무관하게 정렬하여 키 생성)
                    edge_key = tuple(sorted((src_id, dst_id)))
                    
                    if edge_key not in self.sent_edges_cache:
                        edges_to_send.append((src_id, dst_id, score))
                        self.sent_edges_cache.add(edge_key) # 캐시에 추가
                        new_edges_count += 1
            
            # 캐시 메모리 관리 (100만 개 넘으면 초기화)
            if len(self.sent_edges_cache) > 1000000:
                self.sent_edges_cache.clear()
                logger.info("🧹 Edge cache cleared to free up memory.")

            # 새로 발견된 엣지만 전송
            if edges_to_send:
                self.send_edges_batch(edges_to_send)
                logger.info(f"🌱 FAISS: Found {new_edges_count} NEW edges! (Cached: {len(self.sent_edges_cache)}) | Thr: {dynamic_threshold:.2f}")
            else:
                # 중복이라서 보낼 게 없으면 조용히 패스 (로그 공해 방지)
                pass

    def send_edges_batch(self, edges):
        try:
            payload = [{"src": s, "dst": d, "weight": w, "type": "knn_global"} for s, d, w in edges]
            res = requests.post(f"{self.server_url}/add_edges_batch", json=payload, timeout=2.0)
            if not res.ok:
                for item in payload:
                    requests.post(f"{self.server_url}/add_edge", json=item, timeout=0.1)
        except: pass

if __name__ == "__main__":
    gardener = RemoteGardener()
    gardener.run_continuous_training()