export RAID_DIR="<RAID_DIR>"

# Replace PyTorch Lightning progress.py
python -c "import pytorch_lightning as pl; print(pl.__file__)" > pl_path.txt
PL_PATH=$(cat pl_path.txt)
PROGRESS_PY_PATH=$(dirname $PL_PATH)/loops/progress.py
echo "Contents before replacement:"
cat $PROGRESS_PY_PATH
echo "Replacing $PROGRESS_PY_PATH"
cp ${RAID_DIR}/LeanAgent/custom_progress.py $PROGRESS_PY_PATH
echo "Contents after replacement:"
cat $PROGRESS_PY_PATH

# Replace lean_dojo traced_data.py
python -c "import lean_dojo; print(lean_dojo.__file__)" > ld_path.txt
LD_PATH=$(cat ld_path.txt)
TRACED_DATA_PY_PATH=$(dirname $LD_PATH)/data_extraction/traced_data.py
echo "Contents before replacement:"
cat $TRACED_DATA_PY_PATH
echo "Replacing $TRACED_DATA_PY_PATH"
cp ${RAID_DIR}/LeanAgent/custom_traced_data.py $TRACED_DATA_PY_PATH
echo "Contents after replacement:"
cat $TRACED_DATA_PY_PATH

# Replace lean_dojo utils.py
python -c "import lean_dojo; print(lean_dojo.__file__)" > ld_path.txt
LD_PATH=$(cat ld_path.txt)
UTILS_PY_PATH=$(dirname $LD_PATH)/utils.py
echo "Contents before replacement:"
cat $UTILS_PY_PATH
echo "Replacing $UTILS_PY_PATH"
cp ${RAID_DIR}/LeanAgent/custom_utils.py $UTILS_PY_PATH
echo "Contents after replacement:"
cat $UTILS_PY_PATH