FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=9.0

WORKDIR /workspace/TrajFlow

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    build-essential \
    git \
    ninja-build \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY setup/requirements.txt setup/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r setup/requirements.txt

COPY . .

RUN python setup/setup_trajflow.py develop && \
    python -c "import torch; from waymo_open_dataset.metrics.ops import py_metrics_ops; import trajflow.version; import trajflow.mtr_ops.knn.knn_cuda as knn_cuda; import trajflow.mtr_ops.attention.attention_cuda as attention_cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'trajflow', trajflow.version.__version__, 'ops', knn_cuda.__name__, attention_cuda.__name__)"

CMD ["bash"]
