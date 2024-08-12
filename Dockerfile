FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV ROOT_DIR="/workspace"
ENV DATA_DIR="datasets"
ENV CHECKPOINT_DIR="checkpoints"
ENV HUGGINGFACE_API_URL="https://huggingface.co/api/models"
ENV HUGGINGFACE_USER="AK123321"
ENV HUGGINGFACE_TOKEN="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"
ENV PYTHONPATH="${PYTHONPATH}:/workspace/ReProver"
ENV GITHUB_ACCESS_TOKEN="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"
ENV CACHE_DIR="/workspace/.cache/lean_dojo"
ENV ELAN_HOME="/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"
ENV MEGA_EMAIL="adarshk1234567@gmail.com"
ENV MEGA_PASSWORD="0879197abc"

WORKDIR /workspace

RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y && \
    elan toolchain install leanprover/lean4:4.8.0 && \
    elan toolchain install leanprover/lean4:4.8.0-rc2 && \
    elan toolchain install leanprover/lean4:4.8.0-rc1 && \
    elan toolchain install leanprover/lean4:4.7.0 && \
    elan toolchain install leanprover/lean4:4.7.0-rc2 && \
    elan toolchain install leanprover/lean4:4.7.0-rc1 && \
    git clone https://github.com/Adarsh321123/LeanBot.git ReProver && \
    cd ReProver && \
    git checkout backup_branch && \
    rm -rf .git

WORKDIR /workspace/ReProver

RUN ls -la /workspace/ReProver
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get -y install sudo

RUN python -c "import pytorch_lightning as pl; print(pl.__file__)" > pl_path.txt && \
    PL_PATH=$(sudo cat pl_path.txt) && \
    PROGRESS_PY_PATH=$(dirname $PL_PATH)/loops/progress.py && \
    echo "Contents before replacement:" && \
    sudo cat $PROGRESS_PY_PATH && \
    echo "Replacing $PROGRESS_PY_PATH" && \
    sudo cp /workspace/ReProver/custom_progress.py $PROGRESS_PY_PATH && \
    echo "Contents after replacement:" && \
    sudo cat $PROGRESS_PY_PATH

CMD ["python", "/workspace/ReProver/compute_server.py"]

# docker system prune -a
# docker build -t ak123321/leancopilot-compute:latest . && docker push ak123321/leancopilot-compute:latest
# docker build -t ak123321/leancopilot-compute-medium:latest . && docker push ak123321/leancopilot-compute-medium:latest
# docker build -t ak123321/leancopilot-compute-new:latest --no-cache . && docker push ak123321/leancopilot-compute-new:latest
# docker run --rm --gpus all --cpus 9.0 --memory 50g ak123321/leancopilot-compute-new:latest