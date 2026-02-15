import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import HybridQueryGNN

class GraphLLM_Connector(nn.Module):
    def __init__(self, gnn_model, llm_model_id="Qwen/Qwen2.5-1.5B-Instruct", freeze_llm=True):
        super().__init__()
        
        # 1. 선생님의 GNN (뇌)
        self.gnn = gnn_model
        
        # 2. 천재 LLM (입) - 4비트로 로드해서 VRAM 절약
        print(f"🤖 Loading LLM: {llm_model_id}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True # ★ VRAM 4GB로 구동 가능!
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        
        # 3. 연결 다리 (Projector)
        # GNN(768) -> LLM(2560 등) 차원 변환
        llm_hidden_size = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(768, llm_hidden_size),
            nn.ReLU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        # 4. LLM 얼리기 (너는 이미 똑똑하니까 배우지 마)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("❄️ LLM parameters frozen.")

    def forward(self, x, edge_index, query_text_list, batch=None):
        # 1. GNN 실행 (지식 추출)
        # query_text_list는 텍스트 리스트
        query_emb = ... # (텍스트 -> 임베딩 변환 로직 필요)
        h_nodes, h_global, _ = self.gnn(x, edge_index, query_emb, batch)
        
        # 2. 그래프 지식을 LLM 언어로 번역 (Vector Projection)
        # h_global(256) -> graph_tokens(2560)
        graph_tokens = self.projector(h_global).unsqueeze(1) # (Batch, 1, Hidden)
        
        # 3. 텍스트 프롬프트 처리
        # 예: "다음 그래프 정보를 바탕으로 답변해: "
        prompts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in query_text_list]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.llm.device)
        
        input_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
        
        # ★ [핵심] 그래프 벡터를 텍스트 임베딩 앞에 붙임!
        # [Graph Vector] + [User Query Vector]
        combined_embeds = torch.cat([graph_tokens, input_embeds], dim=1)
        
        # 4. LLM에게 넘김
        outputs = self.llm(inputs_embeds=combined_embeds, attention_mask=inputs.attention_mask)
        
        return outputs