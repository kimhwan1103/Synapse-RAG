#!/bin/bash

# ============================================================
# Hybrid GraphDB v2 - 빌드 및 사용 가이드
# ============================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Hybrid GraphDB v2 - Build & Usage Guide              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. 의존성 확인
echo "[1/5] Checking dependencies..."
echo "----------------------------------------"

if ! command -v g++ &> /dev/null; then
    echo "❌ g++ not found! Please install: sudo apt-get install build-essential"
    exit 1
fi

if ! ldconfig -p | grep -q librocksdb; then
    echo "❌ RocksDB not found! Please install:"
    echo "   sudo apt-get install librocksdb-dev"
    exit 1
fi

echo "✅ All dependencies found"
echo ""

# 2. 컴파일
echo "[2/5] Compiling..."
echo "----------------------------------------"

# Maintenance Tool 컴파일
echo "Compiling db_maintenance_tool..."
g++ -o db_maintenance_tool \
    db_maintenance_tool.cpp \
    hybrid_graphDB_v2.cpp \
    -std=c++17 \
    -O3 \
    -lrocksdb \
    -lpthread \
    -ldl \
    -lz \
    -lbz2 \
    -lsnappy \
    -llz4 \
    -lzstd \
    2>&1 | tee compile.log

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
else
    echo "❌ Compilation failed! Check compile.log"
    exit 1
fi

echo ""

# 3. 사용법 안내
echo "[3/5] Usage Guide"
echo "----------------------------------------"
echo ""
echo "🔧 Maintenance Tool:"
echo "   ./db_maintenance_tool [db_path]"
echo ""
echo "   Example: ./db_maintenance_tool /tmp/my_hybrid_db"
echo ""
echo "   Features:"
echo "   1. Show graph statistics"
echo "   2. Validate text quality"
echo "   3. Find orphan nodes"
echo "   4. Build similarity graph"
echo "   5. Build k-NN graph"
echo "   6. Fix orphan nodes"
echo "   7. Export problematic nodes"
echo "   8. Clear cache"
echo ""

# 4. 권장 워크플로우
echo "[4/5] Recommended Workflow"
echo "----------------------------------------"
echo ""
echo "📋 Step 1: Check Current State"
echo "   Run maintenance tool → Option 1 (Statistics)"
echo ""
echo "📋 Step 2: Validate Data Quality"
echo "   Run maintenance tool → Option 2 (Text Quality)"
echo "   Run maintenance tool → Option 3 (Orphan Nodes)"
echo ""
echo "📋 Step 3: Fix Issues"
echo "   If many orphans: Option 6 (Fix Orphans)"
echo "   Or rebuild entire graph: Option 4 or 5"
echo ""
echo "📋 Step 4: Verify"
echo "   Run maintenance tool → Option 1 (Statistics)"
echo "   Check if edges increased significantly"
echo ""

# 5. 성능 가이드
echo "[5/5] Performance Guide"
echo "----------------------------------------"
echo ""
echo "⚡ Fast Methods (minutes):"
echo "   - k-NN Graph (Option 5): k=10, ~5-10 minutes for 100K nodes"
echo "   - Fix Orphans (Option 6): threshold=0.6, sample=500"
echo ""
echo "🐌 Slow Methods (hours):"
echo "   - Similarity Graph (Option 4): threshold=0.7, sample=1000"
echo "     Recommended for initial build only"
echo ""
echo "💡 Tips:"
echo "   - Start with k-NN graph for quick results"
echo "   - Use lower threshold (0.6) for more edges"
echo "   - Batch size 100-200 for balanced speed/memory"
echo "   - Clear cache regularly (Option 8) during long operations"
echo ""

# 6. 예상 시간
echo "📊 Estimated Time (for 100,000 nodes):"
echo "   - Statistics: < 1 minute"
echo "   - Text Validation: ~2 minutes"
echo "   - Find Orphans: ~2 minutes"
echo "   - k-NN Graph (k=10): ~10 minutes"
echo "   - Similarity Graph (sample=1000): ~2-3 hours"
echo "   - Fix Orphans (sample=500): ~5-10 minutes"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Build Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Run: ./db_maintenance_tool /tmp/my_hybrid_db"
echo ""
