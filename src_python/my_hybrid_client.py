import requests
import numpy as np

class HybridDBClient:
    def __init__(self, server_url="http://rag-db-server:9000"):
        self.server_url = server_url

    def get_node_count(self):
        try:
            res = requests.get(f"{self.server_url}/count")
            return res.json()["count"] if res.ok else 0
        except: return 0
    
    def get_node(self, node_id):
        try:
            res = requests.get(f"{self.server_url}/node/{node_id}")
            if res.ok:
                data = res.json()
                class NodeObj:
                    def __init__(self, t, e): self.text= t; self.embedding=e
                return NodeObj(data["text"], data["embedding"])
        except: pass
        return None
    
    def add_node(self, text, embedding):
        if hasattr(embedding, 'tolist'): embedding = embedding.tolist()
        try:
            requests.post(f"{self.server_url}/add", json={"text":text, "embedding":embedding})
        except: pass

    def search(self, vector, k=5):
        if  hasattr(vector, 'tolist'): vector = vector.tolist()
        try:
            res = requests.post(f"{self.server_url}/search", json={"vector":vector, "k":k})
            if res.ok: return res.json()["distances"], res.json()["indices"]
        except: pass
        return [], []