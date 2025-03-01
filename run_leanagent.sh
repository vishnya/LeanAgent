#!/bin/bash
export RAID_DIR="<RAID_DIR>"
cd ${RAID_DIR}/LeanAgent
echo "Script executed from: ${PWD}"
source $<PATH_TO_CONDA_ENV>/etc/profile.d/conda.sh
conda activate LeanAgent
export PYTHONPATH="${PYTHONPATH}:${RAID_DIR}/LeanAgent"
export GITHUB_ACCESS_TOKEN="<GITHUB_ACCESS_TOKEN>"
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
echo "Running leanagent.py"
python leanagent.py
