#!/bin/bash

# ============================================================
# HybridGraphDB v2 - 완전한 빌드 및 설치 스크립트
# ============================================================

set -e  # 에러 발생 시 중단
set -o pipefail # 파이프라인 중간에서 에러 발생 시 중단 (tee 사용 시 필요)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║    HybridGraphDB v2 - Complete Build & Install            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================
# 1. 의존성 확인
# ============================================================

echo -e "${YELLOW}[1/6] Checking dependencies...${NC}"
echo "----------------------------------------"

# CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}❌ CMake not found!${NC}"
    echo "Install: sudo apt-get install cmake"
    exit 1
fi
echo -e "${GREEN}✅ CMake found:${NC} $(cmake --version | head -n1)"

# g++
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}❌ g++ not found!${NC}"
    echo "Install: sudo apt-get install build-essential"
    exit 1
fi
echo -e "${GREEN}✅ g++ found:${NC} $(g++ --version | head -n1)"

# RocksDB
if ! ldconfig -p | grep -q librocksdb; then
    echo -e "${RED}❌ RocksDB not found!${NC}"
    echo "Install: sudo apt-get install librocksdb-dev"
    exit 1
fi
echo -e "${GREEN}✅ RocksDB found${NC}"

# Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found:${NC} $(python3 --version)"

# pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 not found!${NC}"
    echo "Install: sudo apt-get install python3-pip"
    exit 1
fi
echo -e "${GREEN}✅ pip found${NC}"

echo ""

# ============================================================
# 2. Python 의존성 설치
# ============================================================

echo -e "${YELLOW}[2/6] Installing Python dependencies...${NC}"
echo "----------------------------------------"

pip3 install --upgrade pip setuptools wheel
pip3 install pybind11 numpy tqdm

echo -e "${GREEN}✅ Python dependencies installed${NC}"
echo ""

# ============================================================
# 3. C++ Maintenance Tool 빌드
# ============================================================

echo -e "${YELLOW}[3/6] Building C++ Maintenance Tool...${NC}"
echo "----------------------------------------"

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
    2>&1 | tee build_maintenance.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Maintenance tool built successfully${NC}"
else
    echo -e "${RED}❌ Build failed! Check build_maintenance.log${NC}"
    exit 1
fi

echo ""

# ============================================================
# 4. Python 바인딩 빌드
# ============================================================

echo -e "${YELLOW}[4/6] Building Python bindings...${NC}"
echo "----------------------------------------"

# 소스 파일 준비 (필요시)
if [ ! -f "python_bindings.cpp" ] && [ -f "../../src_cpp/bindings.cpp" ]; then
    echo "  Copying bindings.cpp from src_cpp..."
    cp "../../src_cpp/bindings.cpp" "python_bindings.cpp"
    sed -i 's/hybrid_graphDB.hpp/hybrid_graphDB_v2.hpp/g' python_bindings.cpp
    sed -i 's/my_hybrid_backend/hybrid_graphdb_py/g' python_bindings.cpp
fi

# setup.py를 이용한 빌드 (CMake 의존성 제거)
python3 setup.py build_ext --inplace 2>&1 | tee build_python.log

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed! Check build_python.log${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python bindings built successfully${NC}"
echo ""

# ============================================================
# 5. Python 패키지 설치
# ============================================================

echo -e "${YELLOW}[5/6] Installing Python package...${NC}"
echo "----------------------------------------"

pip3 install -e . 2>&1 | tee install.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python package installed${NC}"
else
    echo -e "${RED}❌ Installation failed! Check install.log${NC}"
    exit 1
fi

echo ""

# ============================================================
# 6. 설치 검증
# ============================================================

echo -e "${YELLOW}[6/6] Verifying installation...${NC}"
echo "----------------------------------------"

# C++ tool 확인
if [ -f "./db_maintenance_tool" ]; then
    echo -e "${GREEN}✅ C++ Maintenance Tool: ./db_maintenance_tool${NC}"
else
    echo -e "${RED}❌ Maintenance tool not found${NC}"
fi

# Python 모듈 확인
python3 -c "import hybrid_graphdb_py as hgdb; print(f'✅ Python Module: hybrid_graphdb_py v{hgdb.__version__}')" 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python module import failed${NC}"
    exit 1
fi

# Wiki reparser 확인
if [ -f "./wiki_reparser.py" ]; then
    chmod +x wiki_reparser.py
    echo -e "${GREEN}✅ Wiki Reparser: ./wiki_reparser.py${NC}"
fi

echo ""

# ============================================================
# 7. 사용법 안내
# ============================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  Installation Complete!                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}📦 Installed Components:${NC}"
echo "  1. C++ Maintenance Tool: ./db_maintenance_tool"
echo "  2. Python Module: hybrid_graphdb_py"
echo "  3. Wiki Reparser: ./wiki_reparser.py"
echo ""
echo -e "${GREEN}🚀 Quick Start:${NC}"
echo ""
echo "  ${YELLOW}# 1. 위키피디아 재파싱${NC}"
echo "  python3 wiki_reparser.py -i input.txt -o output.jsonl"
echo ""
echo "  ${YELLOW}# 2. DB에 로딩 (Python)${NC}"
echo "  python3 << EOF"
echo "  import hybrid_graphdb_py as hgdb"
echo "  import json"
echo "  "
echo "  db = hgdb.HybridGraphDB('/tmp/my_db')"
echo "  "
echo "  with open('output.jsonl', 'r') as f:"
echo "      for line in f:"
echo "          chunk = json.loads(line)"
echo "          # 임베딩 생성 필요"
echo "          db.add_node(chunk['chunk_id'], chunk['text'], embedding)"
echo "  EOF"
echo ""
echo "  ${YELLOW}# 3. 그래프 구축${NC}"
echo "  python3 << EOF"
echo "  import hybrid_graphdb_py as hgdb"
echo "  db = hgdb.HybridGraphDB('/tmp/my_db')"
echo "  db.build_knn_graph(k=10)"
echo "  db.print_graph_stats()"
echo "  EOF"
echo ""
echo "  ${YELLOW}# 4. 또는 C++ Tool 사용${NC}"
echo "  ./db_maintenance_tool /tmp/my_db"
echo ""
echo -e "${GREEN}📚 Documentation:${NC}"
echo "  - GUIDE_V2.md: 종합 가이드"
echo "  - README_BINDINGS.md: Python API 문서"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                     Happy Coding! 🎉                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
