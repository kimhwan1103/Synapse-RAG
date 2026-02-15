import my_hybrid_client as myclient
import numpy as np
import sys
import time

def run_diagnostics():
    print("="*50)
    print("[Diagnosis] Starting DB Connection Check...")
    print("="*50)

    #서버 연결 설정
    server_url = "http://rag-db-server:9000"

    try:
        client = myclient.HybridDBClient(server_url=server_url)
    except Exception as e:
        print(f"Client Initialzation Failed: {e}")
        return
    
    #Ping, Count 테스트
    print("\n[1/1] Checking Server Status...")
    try:
        initial_count = client.get_node_count()
        print(f"Connection Success! (Current DB Nodes : {initial_count})")
    except Exception as e:
        print(f"Connection Falied. Is the server running? Error : {e}")
        return
    
if __name__ == "__main__":
    run_diagnostics()