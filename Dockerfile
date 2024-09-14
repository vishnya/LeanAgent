FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV ROOT_DIR="/workspace"
ENV DATA_DIR="datasets"
ENV CHECKPOINT_DIR="checkpoints"
ENV HUGGINGFACE_API_URL="https://huggingface.co/api/models"
ENV HUGGINGFACE_USER="<>"
ENV HUGGINGFACE_TOKEN="<>"
ENV PYTHONPATH="${PYTHONPATH}:/workspace/ReProver"
ENV GITHUB_ACCESS_TOKEN="<>"
ENV CACHE_DIR="/workspace/.cache/lean_dojo"
ENV ELAN_HOME="/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"
ENV MEGA_EMAIL="<>"
ENV MEGA_PASSWORD="<>"

WORKDIR /workspace

RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y && \
    elan toolchain install leanprover/lean4:4.8.0 && \
    elan toolchain install leanprover/lean4:4.8.0-rc2 && \
    elan toolchain install leanprover/lean4:4.8.0-rc1 && \
    elan toolchain install leanprover/lean4:4.7.0 && \
    elan toolchain install leanprover/lean4:4.7.0-rc2 && \
    elan toolchain install leanprover/lean4:4.7.0-rc1 && \
    elan toolchain install leanprover/lean4:4.6.1 && \
    elan toolchain install leanprover/lean4:4.6.0 && \
    elan toolchain install leanprover/lean4:4.6.0-rc1 && \
    elan toolchain install leanprover/lean4:4.5.0 && \
    elan toolchain install leanprover/lean4:4.5.0-rc1 && \
    elan toolchain install leanprover/lean4:4.4.0 && \
    elan toolchain install leanprover/lean4:4.4.0-rc1 && \
    elan toolchain install leanprover/lean4:4.3.0 && \
    elan toolchain install leanprover/lean4:4.3.0-rc2 && \
    elan toolchain install leanprover/lean4:4.3.0-rc1 && \
    
    elan toolchain install leanprover/lean4:v4.8.0 && \
    elan toolchain install leanprover/lean4:v4.8.0-rc2 && \
    elan toolchain install leanprover/lean4:v4.8.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.7.0 && \
    elan toolchain install leanprover/lean4:v4.7.0-rc2 && \
    elan toolchain install leanprover/lean4:v4.7.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.6.1 && \
    elan toolchain install leanprover/lean4:v4.6.0 && \
    elan toolchain install leanprover/lean4:v4.6.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.5.0 && \
    elan toolchain install leanprover/lean4:v4.5.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.4.0 && \
    elan toolchain install leanprover/lean4:v4.4.0-rc1 && \
    elan toolchain install leanprover/lean4:v4.3.0 && \
    elan toolchain install leanprover/lean4:v4.3.0-rc2 && \
    elan toolchain install leanprover/lean4:v4.3.0-rc1 && \
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

RUN python -c "import lean_dojo; print(lean_dojo.__file__)" > ld_path.txt && \
    LD_PATH=$(sudo cat ld_path.txt) && \
    TRACED_DATA_PY_PATH=$(dirname $LD_PATH)/data_extraction/traced_data.py && \
    echo "Contents before replacement:" && \
    sudo cat $TRACED_DATA_PY_PATH && \
    echo "Replacing $TRACED_DATA_PY_PATH" && \
    sudo cp /workspace/ReProver/custom_traced_data.py $TRACED_DATA_PY_PATH && \
    echo "Contents after replacement:" && \
    sudo cat $TRACED_DATA_PY_PATH

RUN python -c "import lean_dojo; print(lean_dojo.__file__)" > ld_path.txt && \
    LD_PATH=$(sudo cat ld_path.txt) && \
    UTILS_PY_PATH=$(dirname $LD_PATH)/utils.py && \
    echo "Contents before replacement:" && \
    sudo cat $UTILS_PY_PATH && \
    echo "Replacing $UTILS_PY_PATH" && \
    sudo cp /workspace/ReProver/custom_utils.py $UTILS_PY_PATH && \
    echo "Contents after replacement:" && \
    sudo cat $UTILS_PY_PATH

CMD ["python", "/workspace/ReProver/compute_server.py"]

# docker system prune -a
# docker build -t ak123321/leancopilot-compute:latest . && docker push ak123321/leancopilot-compute:latest
# docker build -t ak123321/leancopilot-compute-medium:latest . && docker push ak123321/leancopilot-compute-medium:latest
# docker build -t ak123321/leancopilot-compute-new:latest --no-cache . && docker push ak123321/leancopilot-compute-new:latest
# docker run --rm --gpus all --cpus 9.0 --memory 50g ak123321/leancopilot-compute-new:latest