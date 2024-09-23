"""Script for training the premise retriever.
"""

import os
from typing import Tuple
import numpy as np
import pickle
import json
from tqdm import tqdm
from loguru import logger
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
import sys

from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule

class CLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
                         save_config_kwargs={"overwrite": True},
                         **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")
        parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset.')
    
    def before_instantiate_classes(self):
        cur_data_path = vars(vars(self.config)["predict"])["data_path"]
        # Modify the --config YAML file to include the current data_path
        vars(vars(vars(self.config)["predict"])["data"])["data_path"] = cur_data_path + "/random"
        vars(vars(vars(self.config)["predict"])["data"])["corpus_path"] = cur_data_path + "/corpus.jsonl"
        logger.info(f"Data path: {vars(vars(vars(self.config)['predict'])['data'])['data_path']}")
        logger.info(f"Corpus path: {vars(vars(vars(self.config)['predict'])['data'])['corpus_path']}")

# def load_eval_results():
#     R1 = []
#     R10 = []
#     MRR = []
#     file_path = "retrieval/evaluation_results.txt"
#     if os.path.exists(file_path):
#         with open(file_path, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if line.startswith("R1:"):
#                     r1 = float(line.split(": ")[1])
#                     R1.append(r1)
#                 elif line.startswith("R10:"):
#                     r10 = float(line.split(": ")[1])
#                     R10.append(r10)
#                 elif line.startswith("MRR:"):
#                     mrr = float(line.split(": ")[1])
#                     MRR.append(mrr)
#         logger.info(f"R1: {r1}, R10: {r10}, MRR: {mrr}")
#     else:
#         logger.info("No evaluation results found.")
#     return R1, R10, MRR

def run_cli(model_path, data_path):
    logger.info(f"PID: {os.getpid()}")
    # Mimic command line argument passing
    sys.argv = ['main.py', 'predict', '--config', 'retrieval/confs/cli_lean4_random_big_model.yaml', '--ckpt_path', model_path, '--data-path', data_path]
    cli = CLI(PremiseRetriever, RetrievalDataModule)

def main() -> None:
    # parser = argparse.ArgumentParser(
    #     description="Script for evaluating the premise retriever."
    # )

    # parser.add_argument(
    #     "--data-path",
    #     type=str,
    #     required=False,
    #     help="Path to the dataset.",
    # )

    # known_args, remaining_argv = parser.parse_known_args()
    # sys.argv[1:] = remaining_argv
    # global cur_data_path
    # cur_data_path = known_args.data_path
    # logger.info(f"Data path: {known_args.data_path}")

    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
