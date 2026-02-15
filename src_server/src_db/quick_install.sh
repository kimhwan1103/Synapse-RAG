#!/bin/bash
# 간단 빌드 스크립트

set -e

echo "🔨 Building HybridGraphDB Python Bindings..."
echo ""

# 1. 필요한 파일 복사
echo "[1/4] Copying files from outputs..."
# 파일이 이미 현재 디렉토리에 있으므로 cp 명령 제거

# python_bindings.cpp가 없다면 ../../src_cpp/bindings.cpp에서 가져오기
if [ ! -f "python_bindings.cpp" ] && [ -f "../../src_cpp/bindings.cpp" ]; then
    echo "  Copying bindings.cpp from src_cpp..."
    cp "../../src_cpp/bindings.cpp" "python_bindings.cpp"
    # 헤더 파일명 및 모듈명 수정 (v2 호환)
    sed -i 's/hybrid_graphDB.hpp/hybrid_graphDB_v2.hpp/g' python_bindings.cpp
    sed -i 's/my_hybrid_backend/hybrid_graphdb_py/g' python_bindings.cpp
fi

echo "✅ Files copied"
echo ""

# 2. Python 의존성
echo "[2/4] Installing Python dependencies..."
pip3 install pybind11 numpy tqdm sentence-transformers -q

echo "✅ Dependencies installed"
echo ""

# 3. 빌드
echo "[3/4] Building Python module..."
pip3 install -e . 2>&1 | grep -E "(Processing|Successfully|ERROR)" || true

if python3 -c "import hybrid_graphdb_py" 2>/dev/null; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed! Trying alternative method..."
    
    # CMake 방식 대신 g++ 직접 빌드 (RocksDB CMake 설정 문제 회피)
    echo "  Compiling with g++ directly..."
    
    # 확장자 가져오기
    EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
    
    if g++ -O3 -Wall -shared -std=c++17 -fPIC \
        $(python3 -m pybind11 --includes) \
        python_bindings.cpp hybrid_graphDB_v2.cpp \
        -o "hybrid_graphdb_py${EXT_SUFFIX}" \
        -lrocksdb -lpthread -ldl -lz -lbz2 -lsnappy -llz4 -lzstd; then
        echo "✅ Manual build successful!"
    else
        echo "❌ Manual build failed!"
        exit 1
    fi
fi

echo ""

# 4. 검증
echo "[4/4] Verifying installation..."
python3 << 'EOF'
try:
    import hybrid_graphdb_py as hgdb
    print(f"✅ Module loaded: hybrid_graphdb_py v{hgdb.__version__}")
    print("✅ Installation complete!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)
EOF

echo ""
echo "🎉 Ready to use!"
echo ""
echo "Run: python3 wiki_full_pipeline.py --dump your_dump.xml.bz2 --db /tmp/wiki_db"
