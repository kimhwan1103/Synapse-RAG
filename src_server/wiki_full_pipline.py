#!/usr/bin/env python3
"""
위키피디아 완전 자동 처리 파이프라인 v2
- 원본 덤프 파일 직접 처리
- 재파싱 → 임베딩 → DB 로딩 → 그래프 구축
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """의존성 확인"""
    missing = []
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        import hybrid_graphdb_py as hgdb
    except ImportError:
        missing.append("hybrid_graphdb_py (run: ./build_all.sh)")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        logger.error("❌ Missing dependencies:")
        for dep in missing:
            logger.error(f"   - {dep}")
        logger.info("\nInstall with:")
        logger.info("   pip3 install sentence-transformers tqdm")
        logger.info("   ./build_all.sh  # for hybrid_graphdb_py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='위키피디아 완전 자동 처리 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 덤프 파일부터 처리 (완전 자동)
  python wiki_full_pipeline.py \\
      --dump kowiki-latest-pages-articles.xml.bz2 \\
      --db /tmp/wiki_v2_db
  
  # 이미 재파싱된 JSONL 사용
  python wiki_full_pipeline.py \\
      --jsonl wiki_reparsed.jsonl \\
      --db /tmp/wiki_v2_db \\
      --skip-parse

  # 빠른 테스트 (작은 데이터)
  python wiki_full_pipeline.py \\
      --dump small_wiki.xml \\
      --db /tmp/test_db \\
      --max-pages 1000

Download Korean Wikipedia dump:
  wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
  (약 1.2GB 압축, 8GB 해제)
        """
    )
    
    # 입력 소스
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dump', '-d', 
                            help='위키 덤프 파일 (.xml.bz2 또는 .xml)')
    input_group.add_argument('--jsonl', '-j',
                            help='이미 재파싱된 JSONL 파일')
    
    # 출력
    parser.add_argument('--db', '-o', required=True,
                       help='DB 저장 경로')
    
    # 옵션
    parser.add_argument('--embedding-model', '-e',
                       default='jhgan/ko-sroberta-multitask',
                       help='임베딩 모델 (기본: jhgan/ko-sroberta-multitask)')
    parser.add_argument('--skip-parse', action='store_true',
                       help='파싱 스킵 (JSONL이 이미 있을 때)')
    parser.add_argument('--skip-graph', action='store_true',
                       help='그래프 구축 스킵')
    parser.add_argument('--k', type=int, default=10,
                       help='k-NN의 k 값 (기본: 10)')
    parser.add_argument('--max-pages', type=int,
                       help='처리할 최대 페이지 수 (테스트용)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='임베딩 배치 크기 (기본: 100)')
    
    args = parser.parse_args()
    
    # 의존성 확인
    check_dependencies()
    
    # 임포트
    from sentence_transformers import SentenceTransformer
    import hybrid_graphdb_py as hgdb
    from tqdm import tqdm
    
    logger.info("="*60)
    logger.info("🚀 Wikipedia Full Pipeline v2")
    logger.info("="*60)
    
    # ==========================================
    # Step 1: 파싱 (필요시)
    # ==========================================
    
    if args.dump:
        logger.info("[Step 1] Parsing dump file...")
        logger.info("="*60)
        
        jsonl_path = args.db + "_temp.jsonl"
        
        if not args.skip_parse or not os.path.exists(jsonl_path):
            # wiki_dump_parser 실행
            try:
                from wiki_dump_parser import process_wiki_dump
            except ImportError:
                logger.error("❌ wiki_dump_parser.py not found!")
                sys.exit(1)
            
            process_wiki_dump(
                dump_path=args.dump,
                output_path=jsonl_path,
                min_chunk_length=500,
                max_chunk_length=1000
            )
        else:
            logger.info(f"Using existing JSONL: {jsonl_path}")
    else:
        jsonl_path = args.jsonl
        logger.info(f"[Step 1] Using provided JSONL: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        logger.error(f"❌ JSONL file not found: {jsonl_path}")
        sys.exit(1)
    
    # ==========================================
    # Step 2: 임베딩 모델 로드
    # ==========================================
    
    logger.info("\n[Step 2] Loading embedding model...")
    logger.info("="*60)
    
    model = SentenceTransformer(args.embedding_model)
    test_emb = model.encode("테스트")
    embedding_dim = len(test_emb)
    
    logger.info(f"Model: {args.embedding_model}")
    logger.info(f"Dimension: {embedding_dim}")
    logger.info("✅ Model loaded\n")
    
    # ==========================================
    # Step 3: DB 생성 및 데이터 로딩
    # ==========================================
    
    logger.info("[Step 3] Creating database and loading data...")
    logger.info("="*60)
    
    db = hgdb.HybridGraphDB(args.db, threshold=0.7, capacity=5000)
    logger.info(f"DB: {args.db}")
    
    # JSONL 읽기
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    if args.max_pages:
        lines = lines[:args.max_pages]
        logger.info(f"⚠️  Limited to {args.max_pages} chunks for testing")
    
    logger.info(f"Loading {len(lines)} chunks...")
    
    # 배치 처리
    batch_texts = []
    batch_ids = []
    batch_chunks = []
    node_id = 0
    
    for line in tqdm(lines, desc="Loading"):
        chunk = json.loads(line)
        
        batch_texts.append(chunk['text'])
        batch_ids.append(node_id)
        batch_chunks.append(chunk)
        node_id += 1
        
        # 배치가 차면 처리
        if len(batch_texts) >= args.batch_size:
            # 임베딩 생성 (배치)
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            
            # DB에 추가
            for i, (nid, text, emb) in enumerate(zip(batch_ids, batch_texts, embeddings)):
                db.add_node(nid, text, emb.tolist())
            
            batch_texts = []
            batch_ids = []
            batch_chunks = []
        
        # 주기적으로 캐시 정리
        if node_id % 10000 == 0 and node_id > 0:
            db.clear_cache()
    
    # 마지막 배치
    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        for nid, text, emb in zip(batch_ids, batch_texts, embeddings):
            db.add_node(nid, text, emb.tolist())
    
    logger.info(f"✅ Loaded {node_id} nodes\n")
    
    # ==========================================
    # Step 4: 그래프 구축
    # ==========================================
    
    if not args.skip_graph:
        logger.info("[Step 4] Building graph...")
        logger.info("="*60)
        
        logger.info("Before graph building:")
        db.print_graph_stats()
        
        logger.info(f"\nBuilding k-NN graph (k={args.k})...")
        db.build_knn_graph(k=args.k)
        
        logger.info("✅ Graph built\n")
    else:
        logger.info("[Step 4] Skipping graph building")
    
    # ==========================================
    # Step 5: 검증
    # ==========================================
    
    logger.info("[Step 5] Validation...")
    logger.info("="*60)
    
    db.print_graph_stats()
    
    if not args.skip_graph:
        orphans = db.find_orphan_nodes()
        logger.info(f"\nOrphan nodes: {len(orphans)}")
        
        if orphans and len(orphans) < 10000:  # 너무 많으면 스킵
            logger.info("Fixing orphan nodes...")
            db.build_edges_for_nodes(orphans, threshold=0.6, sample_size=500)
            db.print_graph_stats()
    
    # 텍스트 품질
    logger.info("\nValidating text quality...")
    bad_nodes = db.validate_text_quality(min_length=200)
    logger.info(f"Problematic nodes: {len(bad_nodes)}/{db.get_node_count()}")
    
    logger.info("✅ Validation complete\n")
    
    # ==========================================
    # Step 6: 테스트
    # ==========================================
    
    logger.info("[Step 6] Testing...")
    logger.info("="*60)
    
    data = db.get_training_data(batch_size=5)
    
    logger.info(f"Sample batch:")
    logger.info(f"  Nodes: {len(data['nodes'])}")
    logger.info(f"  Edges: {len(data['edges'])}")
    
    if data['nodes']:
        node = data['nodes'][0]
        logger.info(f"\nSample node:")
        logger.info(f"  ID: {node['id']}")
        logger.info(f"  Text: {node['text'][:100]}...")
        logger.info(f"  Embedding dim: {len(node['embedding'])}")
    
    if data['edges']:
        edge = data['edges'][0]
        logger.info(f"\nSample edge:")
        logger.info(f"  {edge['src']} -> {edge['dst']} (weight: {edge['weight']:.3f})")
    
    logger.info("\n✅ Test complete\n")
    
    # ==========================================
    # 완료
    # ==========================================
    
    logger.info("="*60)
    logger.info("✅ Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"Total nodes: {node_id}")
    logger.info(f"DB path: {args.db}")
    if args.dump:
        logger.info(f"Parsed JSONL: {jsonl_path}")
    logger.info("="*60)
    logger.info("\n🎉 Ready for training!\n")
    logger.info("Use in Python:")
    logger.info("  import hybrid_graphdb_py as hgdb")
    logger.info(f"  db = hgdb.HybridGraphDB('{args.db}')")
    logger.info("  data = db.get_training_data(32)")
    logger.info("")


if __name__ == "__main__":
    main()