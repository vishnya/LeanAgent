#!/bin/bash
cd ~/ReProver
echo "Script executed from: ${PWD}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ReProver
export PYTHONPATH="${PYTHONPATH}:~/ReProver"
export GITHUB_ACCESS_TOKEN="<KEY>"
export CACHE_DIR="<DIR>/.cache/lean_dojo"
python ~/ReProver/main.py
