FROM python:3.10-slim

# 2. 시스템 필수 패키지 설치 (RocksDB & 빌드 도구)
# - build-essential, cmake: C++ 확장 모듈 빌드용
# - librocksdb-dev: DB 백엔드용
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    librocksdb-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. Python 라이브러리 설치 (버전 충돌 없는 최신 조합)
# - CPU 전용 PyTorch로 설치하여 이미지 크기 대폭 축소 (GPU 필요시 이 줄 수정)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# - 나머지 추론 필수 라이브러리
RUN pip install --no-cache-dir \
    transformers \
    sentence-transformers \
    fastapi \
    uvicorn \
    requests \
    apscheduler \
    faiss-cpu \
    numpy \
    torch_geomtric

