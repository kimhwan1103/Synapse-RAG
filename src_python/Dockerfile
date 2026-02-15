FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git htop && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch_geometric transformers==4.34.0

WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]