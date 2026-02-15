import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

# ★ [수정 1] Gardener와 Generator가 공유할 '표준 규격' 정의
# 768 -> 1024 -> 768 (정보 손실 없이 고해상도 유지)
class SharedGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels=768, out_channels=768):
        super().__init__()
        # 내부 Hidden을 좀 더 키워서 정보 손실 방지
        self.conv1 = GCNConv(in_channels, 1024)
        self.conv2 = GCNConv(1024, out_channels)

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class HybridQueryGNN(nn.Module):
    def __init__(self, node_dim=768, hidden_dim=768, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # ★ [수정 2] 위에서 만든 표준 규격 사용 (이제 Gardener와 호환됨)
        self.local_gcn = SharedGCNEncoder(node_dim, hidden_dim)
        
        # ★ [수정 3] 안 쓰는 GATv2Conv 제거 (메모리 절약)
        
        # 질문 프로젝션
        self.query_proj = nn.Linear(node_dim, hidden_dim)
        # GRU 업데이트
        self.global_gru = nn.GRUCell(hidden_dim, hidden_dim)
        # 피드백
        self.global_broadcast = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, query_emb, batch=None):
        # 1. Local GCN (Gardener와 뇌 공유)
        h_local = self.local_gcn(x, edge_index) # relu는 encoder 안에 있음
        
        # 2. Global Node Init
        h_global = self.query_proj(query_emb)
        
        # 3. Global Attention Logic
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        h_global_expanded = h_global[batch]
        
        # Attention (Dot Product Attention) - 심플하고 빠름
        # (h_global과 h_local이 같은 차원 256이므로 내적 가능)
        scores = torch.sum(h_global_expanded * h_local, dim=1, keepdim=True)
        # [안정화] 스케일링 추가 (값이 너무 커져서 Softmax가 튀는 것 방지)
        scores = scores / (self.hidden_dim ** 0.5)
        attention_weights = self._softmax(scores, batch)
        
        # Weighted Sum
        weighted_nodes = h_local * attention_weights
        global_message = scatter(weighted_nodes, batch, dim=0, reduce='sum')
        
        # 4. Update Global Context
        h_global_new = self.global_gru(global_message, h_global)
        
        # 5. Feedback to Nodes
        global_context_per_node = self.global_broadcast(h_global_new)[batch]
        h_final = h_local + global_context_per_node
        
        return h_final, h_global_new, attention_weights

    def _softmax(self, src, index):
        out = src - scatter(src, index, dim=0, reduce='max')[index]
        out = out.exp()
        out = out / (scatter(out, index, dim=0, reduce='sum')[index] + 1e-16)
        return out