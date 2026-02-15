#!/usr/bin/env python3
"""
RAG 서버 데이터 확인 스크립트
사용법: python check_rag_data.py
"""

import requests
import json

SERVER_URL = "http://rag-db-server:9000"

def check_data(batch_size=5):
    """서버에서 데이터를 가져와서 상세히 출력"""
    
    print("=" * 80)
    print("🔍 RAG Server Data Inspection")
    print("=" * 80)
    
    try:
        # 데이터 가져오기
        response = requests.get(
            f"{SERVER_URL}/get_training_batch",
            params={"batch_size": batch_size},
            timeout=10
        )
        
        if not response.ok:
            print(f"❌ Server error: {response.status_code}")
            return
        
        data = response.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"\n📊 Summary:")
        print(f"  - Total nodes: {len(nodes)}")
        print(f"  - Total edges: {len(edges)}")
        
        # 노드 상세 정보
        print("\n" + "=" * 80)
        print("📝 Node Details:")
        print("=" * 80)
        
        for i, node in enumerate(nodes, 1):
            node_id = node.get('id', 'N/A')
            text = node.get('text', '')
            embedding = node.get('embedding', [])
            
            print(f"\n[Node {i}]")
            print(f"  ID: {node_id}")
            print(f"  Embedding dim: {len(embedding)}")
            print(f"  Text length: {len(text)} characters")
            print(f"  Text content:")
            print(f"  ┌{'─' * 76}┐")
            
            # 텍스트를 줄바꿈하여 출력
            lines = text.split('\n')
            for line in lines[:10]:  # 처음 10줄만
                if line:
                    # 76자로 자르기
                    if len(line) > 76:
                        print(f"  │ {line[:76]}")
                    else:
                        print(f"  │ {line:<76}")
            
            if len(lines) > 10:
                print(f"  │ ... (총 {len(lines)}줄) ...")
                print(f"  │")
                # 마지막 몇 줄도 보여주기
                for line in lines[-3:]:
                    if line:
                        if len(line) > 76:
                            print(f"  │ {line[:76]}")
                        else:
                            print(f"  │ {line:<76}")
            
            print(f"  └{'─' * 76}┘")
            
            # 텍스트가 잘린 것 같은지 확인
            if text.endswith('...') or text.endswith('…'):
                print(f"  ⚠️  WARNING: Text appears to be truncated (ends with ...)")
            
            if not text.endswith(('.', '!', '?', '"', "'", ')', ']', '}', '。', '、')):
                print(f"  ⚠️  WARNING: Text may be incomplete (doesn't end with proper punctuation)")
        
        # 엣지 정보
        print("\n" + "=" * 80)
        print("🔗 Edge Details:")
        print("=" * 80)
        
        if edges:
            print(f"\n  Total edges: {len(edges)}")
            print(f"  Sample edges (first 5):")
            for i, edge in enumerate(edges[:5], 1):
                print(f"    {i}. {edge.get('src')} → {edge.get('dst')}")
        else:
            print("  ⚠️  No edges found!")
        
        # 데이터 품질 체크
        print("\n" + "=" * 80)
        print("🔍 Data Quality Check:")
        print("=" * 80)
        
        valid_nodes = [n for n in nodes if len(n.get('embedding', [])) == 768 and len(n.get('text', '').strip()) > 5]
        
        print(f"\n  Valid nodes (768dim + text>5): {len(valid_nodes)}/{len(nodes)}")
        
        if len(valid_nodes) < len(nodes):
            print(f"  ⚠️  {len(nodes) - len(valid_nodes)} nodes filtered out!")
            
            # 필터링된 노드 보기
            filtered = [n for n in nodes if n not in valid_nodes]
            for node in filtered:
                text_len = len(node.get('text', ''))
                emb_len = len(node.get('embedding', []))
                print(f"     - Node {node.get('id')}: text={text_len} chars, embedding={emb_len}dim")
        
        # 텍스트 길이 분포
        text_lengths = [len(n.get('text', '')) for n in nodes]
        if text_lengths:
            print(f"\n  Text length statistics:")
            print(f"    - Min: {min(text_lengths)} chars")
            print(f"    - Max: {max(text_lengths)} chars")
            print(f"    - Avg: {sum(text_lengths)/len(text_lengths):.1f} chars")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    batch_size = 5
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except:
            pass
    
    print(f"\nFetching {batch_size} nodes from RAG server...")
    check_data(batch_size)
    print("\n✅ Done!\n")