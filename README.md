# LeanAgent

This repository holds the (messy) code for the associated [paper](https://arxiv.org/abs/2410.06209).

Ensure you are on the `refactor` branch. We would like to clean the code up before merging into `main`.

## Brief Documentation

Note: this is a fork of the [ReProver](https://github.com/lean-dojo/ReProver) repository.

- `generator/`: Contains the scripts to train ReProver's tactic generator.
- `prover/proof_search.py`: This is the code that does the proof search.
- `retrieval/`: Contains the scripts to train ReProver's premise retriever.
- `retrieval/datamodule.py`: Creates datasets and datamodules for models.
- `retrieval/main.py`: CLI interface used in `main.py` to evaluate the retriever after progressive training.
- `common.py`: Various helper functions.
- `compute_fisher.py`: Takes the most recent checkpoint and computes the Fisher Information Matrix for the next run's EWC. While not directly used in LeanAgent, this is useful for the comparison experiments.
- `dynamic_database.py`: Definition of the dynamic database that track's LeanAgent's knowledge over time.
- `unittest_dynamic_database.py`: Various unit tests for the dynamic database.
- `generate_benchmark_lean4.py`: Generates a dataset that can then be placed into the dynamic database.
- `main3.py`: This is the driver code. Please see the commments to better understand what it does. To use it, change any parameters you would like in the `main()` function.
- `run_code.sh`: Convenience script to set up the environment before calling `main.py`. Make sure to change the necessary details, such as your GitHub access token and directory to cache the Lean repos, before running it.

There are many variants of these files in the repository, such as modifications of `main3.py` for different experimental setups, but the core idea is the same.

Do this
In `run_leanagent.sh`
1. Change `RAID_DIR`
2. Install `conda` and change path in `source` command
3. `conda create -n "LeanAgent" python=3.10 && conda activate LeanAgent && pip install -r requirements.txt`
4. Add `GITHUB_ACCESS_TOKEN`

In `leanagent.py`
1. Change and create the globals at the top of the file, including `repo_dir`, `DATA_DIR`, `CHECKPOINT_DIR`, `EVAL_RESULTS_FILE_PATH`, `DB_FILE_NAME`, `PROOF_LOG_FILE_NAME`, `ENCOUNTERED_THEOREMS_FILE`
2. Change any of the options at the top of `main()` as you see fit.
3. To get ReProver's tactic generator, run `pip install gdown` and then `gdown https://drive.google.com/uc?id=1aRd1jQPu_TX15Ib5htzn3wZqHYowl3ax`. Move it to `f"{RAID_DIR}/model_lightning.ckpt"`
4. To get ReProver's starting retriever, run `gdown https://drive.google.com/uc?id=1aRd1jQPu_TX15Ib5htzn3wZqHYowl3ax`. Move it to `f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"`

(optional) In `compute_fisher.py`
1. Change and create `FISHER_DIR` and `new_data_path`
2. To get the starting FIM for mathlib, run `gdown https://drive.google.com/uc?id=1Q8yHq7XTAaHXGCiCGmhwZhTfkhHhN1cP`. Move it to `FISHER_DIR`.

TODO: unless we move to mian, we alos need to set lean dirs in generate benchmark

TODO: ask to install lean through elan as leandojo does

Then, `bash run_leanagent.sh`. 