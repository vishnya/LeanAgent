"""
This code will initialize LeanBot by loading the original ReProver model checkpoint and
the LeanDojo Benchmark 4.
"""
import generate_benchmark_lean4_original
import minimal
import subprocess
import os
import torch
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule
from loguru import logger
import pickle

# TODO: duplication with original generation here and then 2% elsewhere

URL = "https://github.com/leanprover-community/mathlib4"
COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
RAID_DIR = "/raid/adarsh"
DATA_DIR = "datasets"
CHECKPOINTS_DIR = "checkpoints"

def main():
    # Load the original ReProver model checkpoint
    old_directory = RAID_DIR + "/leandojo-pl-ckpts"
    print(f"Current working directory: {os.getcwd()}")
    os.chdir(old_directory)
    print(f"Current working directory: {os.getcwd()}")
    dir_name = URL.split("/")[-1] + "_" + COMMIT + ".ckpt"
    subprocess.run(["python", "retriever_random.ckpt/zero_to_fp32.py", "retriever_random.ckpt/", dir_name], check=True)
    print(f"Checkpoint successfully converted to fp32")
    model_checkpoint_path = old_directory + "/" + dir_name
    minimal.main(model_checkpoint_path)
    print(f"Successfully modified checkpoint")
    new_directory = RAID_DIR + "/" + CHECKPOINTS_DIR
    subprocess.run(["mv", dir_name, new_directory], check=True)
    print(f"Checkpoint moved to new directory")

    # Generate the original benchmark
    # TODO: test this
    dir_name = URL.split("/")[-1] + "_" + COMMIT
    dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name
    generate_benchmark_lean4_original.main(URL, COMMIT, dst_dir)

    # Generate the Fisher Information Matrix for the original ReProver model checkpoint
    # TODO: test this
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "max_seq_len": 512,
        "num_retrieved": 100,
    }
    model = PremiseRetriever.load(
        model_checkpoint_path, device, freeze=False, config=config
    )
    logger.info(f"Loaded premise retriever at {model_checkpoint_path}")
    corpus_path = dst_dir + "/corpus.jsonl"
    data_path = dst_dir + "/random"
    data_module = RetrievalDataModule(
        data_path=data_path,
        corpus_path=corpus_path,
        num_negatives=3,
        num_in_file_negatives=1,
        model_name="google/byt5-small",
        batch_size=4,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=4
    )
    train_dataloader = data_module.train_dataloader()
    fisher_info = model.compute_fisher_information(train_dataloader)
    fisher_name = dir_name + "_fisher_info.pkl"
    with open(fisher_name, "wb") as f:
        pickle.dump(fisher_info, f)
    logger.info(f"Fisher info saved to {fisher_name}")

if __name__ == "__main__":
    main()
