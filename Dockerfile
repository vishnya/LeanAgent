# ARG BASE_IMAGE=runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# FROM ${BASE_IMAGE}

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ENV DEBIAN_FRONTEND=noninteractive
# ENV SHELL=/bin/bash
# ENV ROOT_DIR="/workspace"
# ENV DATA_DIR="datasets"
# ENV CHECKPOINT_DIR="checkpoints"
# ENV HUGGINGFACE_API_URL="https://huggingface.co/api/models"
# ENV USER="AK123321"
# ENV HUGGINGFACE_TOKEN="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"
# ENV PYTHONPATH="${PYTHONPATH}:/workspace/ReProver"
# ENV GITHUB_ACCESS_TOKEN="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"
# ENV CACHE_DIR="/workspace/.cache/lean_dojo"
# ENV ELAN_HOME="/.elan"
# ENV PATH="${ELAN_HOME}/bin:${PATH}"

# WORKDIR /workspace

# RUN apt-get update && \
#     apt-get install -y wget gnupg && \
#     wget -O - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - && \
#     wget -O - https://archive.ubuntu.com/ubuntu/project/ubuntu-archive-keyring.gpg | apt-key add - && \
#     apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C && \
#     apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#     echo "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse" > /etc/apt/sources.list && \
#     echo "deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#     echo "deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
#     echo "deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
#     echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" >> /etc/apt/sources.list && \
#     apt-get update && \
#     apt-get install -y git wget curl bash libgl1 software-properties-common && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y
# RUN elan toolchain install leanprover/lean4:4.10.0 && \
#     elan toolchain install leanprover/lean4:4.9.0 && \
#     elan toolchain install leanprover/lean4:4.8.0-rc1 && \
#     elan toolchain install leanprover/lean4:4.7.0

# RUN git clone https://github.com/Adarsh321123/LeanBot.git ReProver
# WORKDIR /workspace/ReProver
# RUN git checkout backup_branch

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY compute_server.py .
# COPY generate_benchmark_lean4.py .
# COPY common.py .
# COPY scripts ./scripts
# COPY retrieval ./retrieval
# COPY generator ./generator
# COPY prover ./prover

# CMD ["python", "compute_server.py"]

FROM ubuntu:22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV ROOT_DIR="/workspace"
ENV DATA_DIR="datasets"
ENV CHECKPOINT_DIR="checkpoints"
ENV HUGGINGFACE_API_URL="https://huggingface.co/api/models"
ENV USER="AK123321"
ENV HUGGINGFACE_TOKEN="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"
ENV PYTHONPATH="${PYTHONPATH}:/workspace/ReProver"
ENV GITHUB_ACCESS_TOKEN="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"
ENV CACHE_DIR="/workspace/.cache/lean_dojo"
ENV ELAN_HOME="/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y wget curl git software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y
RUN elan toolchain install leanprover/lean4:4.10.0 && \
    elan toolchain install leanprover/lean4:4.9.0 && \
    elan toolchain install leanprover/lean4:4.8.0-rc1 && \
    elan toolchain install leanprover/lean4:4.7.0

RUN git clone https://github.com/Adarsh321123/LeanBot.git ReProver
WORKDIR /workspace/ReProver
RUN git checkout backup_branch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY compute_server.py .
COPY generate_benchmark_lean4.py .
COPY common.py .
COPY scripts ./scripts
COPY retrieval ./retrieval
COPY generator ./generator
COPY prover ./prover

CMD ["python", "compute_server.py"]