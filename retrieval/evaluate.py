"""Script for evaluating the premise retriever."""

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from loguru import logger


def _eval(data, preds_map) -> Tuple[float, float, float]:
    R1 = []
    R10 = []
    MRR = []

    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            pred = preds_map[
                (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            ]
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
    parser = argparse.ArgumentParser(
        description="Script for evaluating the premise retriever."
    )
    parser.add_argument(
        "--preds-files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the retriever's predictions files.",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing the train/val/test splits.",
    )
    args = parser.parse_args()
    logger.info(args)

    total_R1, total_R10, total_MRR = [], [], []
    for preds_file, data_path in zip(args.preds_files, args.data_paths):
        # TODO: maybe later just do test?
        logger.info(f"Loading predictions from {preds_file}")
        preds = pickle.load(open(preds_file, "rb"))
        preds_map = {
            (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
            for p in preds
        }
        assert len(preds) == len(preds_map), "Duplicate predictions found!"
        for split in ("train", "val", "test"): # TODO: why all 3? isn't that a data leakage?
            data_path = os.path.join(args.data_path, f"{split}.json")
            data = json.load(open(data_path))
            logger.info(f"Evaluating on {data_path}")
            R1, R10, MRR = _eval(data, preds_map)
            logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
            total_R1.append(R1)
            total_R10.append(R10)
            total_MRR.append(MRR)

    avg_R1 = np.mean(total_R1)
    avg_R10 = np.mean(total_R10)
    avg_MRR = np.mean(total_MRR)
    logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")


if __name__ == "__main__":
    main()
