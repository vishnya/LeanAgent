"""Script for training the premise retriever.
"""

import os
from typing import Tuple
import numpy as np
import pickle
import json
from tqdm import tqdm
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


def _eval(data, preds_map) -> Tuple[float, float, float]:
        R1 = []
        R10 = []
        MRR = []

        for thm in tqdm(data):
            for i, _ in enumerate(thm["traced_tactics"]):
                # logger.info(f"thm['file_path']: {thm['file_path']}")
                # logger.info(f"thm['full_name']: {thm['full_name']}")
                # logger.info(f"tuple(thm['start']): {tuple(thm['start'])}")
                # logger.info(f"i: {i}")
                # pred = preds_map[
                #     (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
                # ]
                pred = None
                key = (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
                logger.info(f"Checking if key {key} is in preds_map")
                if key in preds_map:
                    pred = preds_map[key]
                    logger.info(f"Key {key} found in predictions")
                else:
                    logger.info(f"Key {key} not found in predictions.")
                    continue  # or handle as appropriate
                all_pos_premises = set(pred["all_pos_premises"])
                if len(all_pos_premises) == 0:
                    continue

                retrieved_premises = pred["retrieved_premises"]
                TP1 = retrieved_premises[0] in all_pos_premises
                R1.append(float(TP1) / len(all_pos_premises))
                TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
                R10.append(float(TP10) / len(all_pos_premises))

                for j, p in enumerate(retrieved_premises):
                    if p in all_pos_premises:
                        MRR.append(1.0 / (j + 1))
                        break
                else:
                    MRR.append(0.0)

        R1 = 100 * np.mean(R1)
        R10 = 100 * np.mean(R10)
        MRR = np.mean(MRR)
        return R1, R10, MRR

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

        # load the text file too
        # preds = json.load(open("test_predictions.txt"))
        # logger.info("Loaded the predictions text file")
        # preds_map = ''
        # with open('test_predictions.txt','r') as f:
        #     for i in f.readlines():
        #         preds_map=i
        # preds_map = eval(preds_map)
        # logger.info("Loaded the predictions text file")
        num_gpus = 4
        preds_map = {}
        for gpu_id in range(num_gpus):
            with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
                preds = pickle.load(f)
                preds_map.update(preds)
    

        # preds_map = pickle.load(open("test_pickle.pkl", "rb"))
        logger.info("Loaded the predictions pickle file")
        # preds_map = {
        #     (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
        #     for p in preds
        # }
        # assert len(preds) == len(preds_map), "Duplicate predictions found!"
        data_path = os.path.join(cur_data_path, "random", "test.json")
        data = json.load(open(data_path))
        logger.info(f"Evaluating on {data_path}")
        R1, R10, MRR = _eval(data, preds_map)
        logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
        # R1, R10, MRR = load_eval_results()
        total_R1.append(R1)
        total_R10.append(R10)
        total_MRR.append(MRR)
    
    avg_R1 = np.mean(total_R1)
    avg_R10 = np.mean(total_R10)
    avg_MRR = np.mean(total_MRR)

    logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

    # Save average accuracies to a file
    file_path = "total_evaluation_results.txt"
    with open(file_path, "w") as f:
        f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")


if __name__ == "__main__":
    main()
