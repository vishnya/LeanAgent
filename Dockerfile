FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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

# Install Elan and Lean
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y && \
    elan toolchain install leanprover/lean4:4.8.0 && \
    elan toolchain install leanprover/lean4:v4.8.0 && \
    elan toolchain install leanprover/lean4:4.8.0-rc2 && \
    elan toolchain install leanprover/lean4:v4.8.0-rc2 && \
    elan toolchain install leanprover/lean4:4.8.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.8.0-rc1 && \
    elan toolchain install leanprover/lean4:4.7.0 && \
    elan toolchain install leanprover/lean4:v4.7.0 && \
    elan toolchain install leanprover/lean4:4.7.0-rc2 && \
    elan toolchain install leanprover/lean4:v4.7.0-rc2 && \
    elan toolchain install leanprover/lean4:4.7.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.7.0-rc1 && \

RUN git clone https://github.com/Adarsh321123/LeanBot.git ReProver && \
    cd ReProver && \
    git checkout backup_branch

WORKDIR /workspace/ReProver

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