# LeanAgent Fisher Information Matrix Computation Script
: <<'
## Overview
This bash script sets up the environment and executes the `compute_fisher.py` script for computing Fisher Information Matrix within the LeanAgent project.

## Prerequisites
- Conda environment named "LeanAgent"
- RAID directory with LeanAgent code
- GitHub access token
- NVIDIA GPU (script checks GPU status)

## Script Actions
1. Sets environment variables including RAID directory path
2. Changes to the LeanAgent directory
3. Activates the LeanAgent conda environment
4. Configures Python path to include LeanAgent
5. Sets GitHub access token for potential API calls
6. Configures cache directory for lean_dojo
7. Performs cleanup:
    - Removes Ray temporary files
    - Creates a custom Ray temporary directory
    - Stops any running Ray processes
    - Kills all Python processes to free memory
8. Displays GPU status via nvidia-smi
9. Executes the compute_fisher.py script

## Usage
Update the placeholder variables before running:
- `<RAID_DIR>`: Path to RAID directory containing LeanAgent
- `<PATH_TO_CONDA_ENV>`: Path to conda installation
- `<GITHUB_ACCESS_TOKEN>`: Your GitHub access token

Then execute: `./run_compute_fisher.sh`
'
#!/bin/bash
export RAID_DIR="<RAID_DIR>"
cd ${RAID_DIR}/LeanAgent
echo "Script executed from: ${PWD}"
source <PATH_TO_CONDA_ENV>/etc/profile.d/conda.sh
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
echo "Killing all python processes to free up memory"
ps aux | grep "python" | awk '{print $2}' | xargs -r kill -9
sleep 5
nvidia-smi
echo "Running compute_fisher.py"
python compute_fisher.py
