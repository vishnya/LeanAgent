#!/bin/bash
cd /home/adarsh/ReProver
echo "Script executed from: ${PWD}"
source /home/adarsh/miniconda3/etc/profile.d/conda.sh
conda activate ReProver
export PYTHONPATH="${PYTHONPATH}:/home/adarsh/ReProver"
export GITHUB_ACCESS_TOKEN="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"
export CACHE_DIR="/raid/adarsh/.cache/lean_dojo"
python /home/adarsh/ReProver/main.py
