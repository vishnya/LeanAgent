#!/bin/bash
cd /home/adarsh/ReProver
echo "Script executed from: ${PWD}"
source /home/adarsh/miniconda3/etc/profile.d/conda.sh
conda activate ReProver
export PYTHONPATH="${PYTHONPATH}:/home/adarsh/ReProver"
export GITHUB_ACCESS_TOKEN="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"
export CACHE_DIR="/raid/adarsh/.cache/lean_dojo"
python /home/adarsh/ReProver/main.py
