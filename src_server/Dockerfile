FROM python:3.9-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    librocksdb-dev \
    libsnappy-dev \
    zlib1g-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    numpy \
    faiss-cpu \
    apscheduler \
    torch \
    torch_geometric \
    pybind11 \
    setuptools \
    wheel

WORKDIR /app

COPY src_cpp /app/src_cpp

WORKDIR /app/src_cpp
RUN pip install .

COPY src_server /app/src_server
COPY src_python/metadata_manager.py /app/metadata_manager.py


CMD ["python3", "src_server/db_server_v2.py"]