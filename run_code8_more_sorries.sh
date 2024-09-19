#!/bin/bash
export RAID_DIR="/data/yingzi_ma/lean_project"
cd ${RAID_DIR}/ReProver
echo "Script executed from: ${PWD}"
source ${RAID_DIR}/../miniconda3/etc/profile.d/conda.sh
conda activate ReProver
export PYTHONPATH="${PYTHONPATH}:${RAID_DIR}/ReProver"
export GITHUB_ACCESS_TOKEN="<>"
export CACHE_DIR="${RAID_DIR}/.cache/lean_dojo"
echo "Removing old cache files"
rm -rf /tmp/ray
echo "Stopping ray"
ray stop --force
echo "Running main8_more_sorries.py"
python main8_more_sorries.py
