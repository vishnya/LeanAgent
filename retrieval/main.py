"""Script for training the premise retriever.
"""

import os
import numpy as np
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule

cur_data_path = None

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")
    
    def before_instantiate_classes(self):
        # Modify the --config YAML file to include the current data_path
        vars(vars(vars(self.config)["predict"])["data"])["data_path"] = cur_data_path + "/random"
        vars(vars(vars(self.config)["predict"])["data"])["corpus_path"] = cur_data_path + "/corpus.jsonl"
        logger.info(f"Data path: {vars(vars(vars(self.config)['predict'])['data'])['data_path']}")
        logger.info(f"Corpus path: {vars(vars(vars(self.config)['predict'])['data'])['corpus_path']}")

def load_eval_results():
    R1 = []
    R10 = []
    MRR = []
    file_path = "retrieval/evaluation_results.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("R1:"):
                    r1 = float(line.split(": ")[1])
                    R1.append(r1)
                elif line.startswith("R10:"):
                    r10 = float(line.split(": ")[1])
                    R10.append(r10)
                elif line.startswith("MRR:"):
                    mrr = float(line.split(": ")[1])
                    MRR.append(mrr)
        logger.info(f"R1: {r1}, R10: {r10}, MRR: {mrr}")
    else:
        logger.info("No evaluation results found.")
    return R1, R10, MRR

# TODO: combine with evaluate.py later
def main() -> None:
    logger.info("Starting tests")
    total_R1, total_R10, total_MRR = [], [], []
    testing_paths = [os.path.join('testing_datasets', d) for d in os.listdir('testing_datasets')]
    for data_path in testing_paths:
        global cur_data_path
        cur_data_path = data_path
        logger.info(f"PID: {os.getpid()}")
        cli = CLI(PremiseRetriever, RetrievalDataModule)
        logger.info("Configuration: \n", cli.config)
        R1, R10, MRR = load_eval_results()
        total_R1.extend(R1)
        total_R10.extend(R10)
        total_MRR.extend(MRR)
    
    avg_R1 = np.mean(total_R1)
    avg_R10 = np.mean(total_R10)
    avg_MRR = np.mean(total_MRR)

    logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

    # Save average accuracies to a file
    file_path = "retrieval/total_evaluation_results.txt"
    with open(file_path, "w") as f:
        f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")


if __name__ == "__main__":
    main()
