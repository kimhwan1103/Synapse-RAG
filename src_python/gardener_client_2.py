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
        
        # Normalize 후 3배 스케일링 (Max Norm = 3.0)
        return F.normalize(x, p=2, dim=1) * 3.0

# ==============================================================================
# 🌿 Gardener Client (Phase 2)
# ==============================================================================
class RemoteGardener:
    def __init__(self, server_url="http://rag-db-server:9000"): 
        self.server_url = server_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화 (768 -> 128)
        self.model = GAE(GraphSAGEEncoder(768, 128)).to(self.device)
        self.load_weights()
        
        # 옵티마이저 & 스케줄러 (재시작 시 LR 리셋)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6
        )
        
        # FAISS 관련 변수
        self.reference_nodes = []
        self.faiss_index = None
        self.is_updating_index = False 
        
        # 중복 전송 방지용 캐시
        self.sent_edges_cache = set()
        
        logger.info(f"🌿 Gardener initialized on {self.device} (Phase 2: Semantic Deep Linking)")

    def load_weights(self):
        weight_file = os.path.join(WEIGHT_PATH, "gardener_encoder_latest.pth")
        if os.path.exists(weight_file):
            try:
                state_dict = torch.load(weight_file, map_location=self.device)
                self.model.encoder.load_state_dict(state_dict)
                logger.info("📂 Loaded existing weights. Proceeding with Phase 2 training.")
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
                    # Phase 2에서는 이미 엣지가 많으므로 신뢰도 높은 것만 학습 (0.1 이상)
                    if e.get("weight", 1.0) > 0.1: 
                        u, v = id_to_idx[e["src"]], id_to_idx[e["dst"]]
                        src_list.append(u)
                        dst_list.append(v)
                        src_list.append(v)
                        dst_list.append(u)
            
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
            # 1. 데이터 가져오기 (768차원)
            res = requests.get(f"{self.server_url}/get_embeddings_bulk", params={"limit": max_nodes}, timeout=180)
            if not res.ok: 
                self.is_updating_index = False
                return

            data = res.json()
            nodes = data.get("nodes", [])
            
            valid_nodes = [n for n in nodes if isinstance(n.get("embedding"), list) and len(n["embedding"]) == 768]
            if not valid_nodes:
                self.is_updating_index = False
                return

            # 2. ★ 모델 통과 (768 -> 128 변환)
            # Global Node들을 모델에 넣어 '의미 벡터(z)'를 뽑아냅니다.
            raw_x = torch.tensor([n["embedding"] for n in valid_nodes], dtype=torch.float).to(self.device)
            
            self.model.eval() # 추론 모드 (BatchNorm 고정)
            with torch.no_grad():
                # 인덱싱 단계에서는 엣지 정보 없이 노드 자체의 의미만 변환합니다.
                # (GraphSAGE는 엣지가 없으면 자기 자신(Self-loop) 정보만으로 인코딩함)
                empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                z_global = self.model.encoder(raw_x, empty_edge_index)
            
            self.model.train() # 다시 학습 모드

            embeddings_128 = z_global.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(embeddings_128) # 코사인 유사도를 위한 정규화
            
            # 3. 인덱스 생성 (128차원)
            index = faiss.IndexFlatIP(128)
            if torch.cuda.is_available():
                res_gpu = faiss.StandardGpuResources()
                res_gpu.setTempMemory(256 * 1024 * 1024) 
                index = faiss.index_cpu_to_gpu(res_gpu, 0, index)
            
            index.add(embeddings_128)
            
            self.faiss_index = index
            self.reference_nodes = [n["id"] for n in valid_nodes]
            
            logger.info(f"✅ [Background] FAISS Index Updated (Semantic 128-dim). Nodes: {len(valid_nodes)}")

        except Exception as e:
            logger.error(f"⚠️ FAISS Update Failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.is_updating_index = False

    def run_continuous_training(self, steps=1000000):
        self.model.train()
        logger.info("🚀 Gardener Phase 2 Started...")
        
        self.update_global_faiss_index()

        for step in range(1, steps + 1):
            data, idx_to_id = self.fetch_batch_data(batch_size=2048)
            
            if not data:
                time.sleep(5)
                continue

            try:
                self.optimizer.zero_grad()
                z = self.model.encode(data.x, data.edge_index)
                
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

                # ★ Phase 2 핵심: 학습된 z(128차원)를 사용하여 검색
                if step % 10 == 0 and self.faiss_index is not None and not self.is_updating_index:
                    self.perform_faiss_linking(z, idx_to_id)
            
            except Exception as e:
                logger.error(f"Step failed: {e}")
                time.sleep(1)
    
    def perform_faiss_linking(self, z, idx_to_id): # 인자 z (128차원)
        with torch.no_grad():
            queries = z.detach().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(queries) # 정규화
            
            k = 5 
            D, I = self.faiss_index.search(queries, k)
            
            avg_score = np.mean(D)
            std_score = np.std(D)
            
            # Phase 2 Threshold: 의미적 유사도는 텍스트보다 점수가 낮을 수 있으므로 유동적으로 조정
            # 하지만 너무 낮으면 엉뚱한게 연결되므로 최소 0.85 유지
            dynamic_threshold = min(0.98, max(0.85, avg_score + 2 * std_score))
            
            if np.max(D) < 0.8: return

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

                    edge_key = tuple(sorted((src_id, dst_id)))
                    
                    if edge_key not in self.sent_edges_cache:
                        edges_to_send.append((src_id, dst_id, score))
                        self.sent_edges_cache.add(edge_key)
                        new_edges_count += 1
            
            if len(self.sent_edges_cache) > 1000000:
                self.sent_edges_cache.clear()
                logger.info("🧹 Edge cache cleared.")

            if edges_to_send:
                self.send_edges_batch(edges_to_send)
                logger.info(f"🧠 Semantic Link: Found {new_edges_count} NEW edges! | Thr: {dynamic_threshold:.2f}")

    def send_edges_batch(self, edges):
        try:
            payload = [{"src": s, "dst": d, "weight": w, "type": "knn_semantic"} for s, d, w in edges]
            res = requests.post(f"{self.server_url}/add_edges_batch", json=payload, timeout=2.0)
            if not res.ok:
                for item in payload:
                    requests.post(f"{self.server_url}/add_edge", json=item, timeout=0.1)
        except: pass

if __name__ == "__main__":
    gardener = RemoteGardener()
    gardener.run_continuous_training()