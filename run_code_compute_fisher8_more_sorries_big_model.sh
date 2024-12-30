#!/bin/bash
export RAID_DIR="/data/yingzi_ma/lean_project"
cd ${RAID_DIR}/ReProver
echo "Script executed from: ${PWD}"
source ${RAID_DIR}/../miniconda3/etc/profile.d/conda.sh
conda activate ReProver
export PYTHONPATH="${PYTHONPATH}:${RAID_DIR}/ReProver"
export GITHUB_ACCESS_TOKEN="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"
export CACHE_DIR="${RAID_DIR}/.cache/lean_dojo"
echo "Removing old cache files"
rm -rf /tmp/ray
export RAY_TMPDIR="${RAID_DIR}/tmp"
rm -rf ${RAY_TMPDIR}
mkdir "${RAY_TMPDIR}"
echo "Stopping ray"
ray stop --force
ps aux | grep "python" | awk '{print $2}' | xargs -r kill -9
sleep 5
nvidia-smi
echo "Running compute_fisher8_more_sorries_big_model.py"
python compute_fisher8_more_sorries_big_model.py
