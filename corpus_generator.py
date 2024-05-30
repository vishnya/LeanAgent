import json
import shutil
import random
import networkx as nx
from copy import copy
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union

import lean_dojo
from lean_dojo import *
from lean_dojo.constants import LEAN4_PACKAGES_DIR

random.seed(3407)  # https://arxiv.org/abs/2109.08203

URL = "https://github.com/Adarsh321123/new-version-test"
COMMIT = "3508f1f7d21f7c31ec7f472d0f5af026971661b8"
DST_DIR = Path("data")
NUM_VAL = NUM_TEST = 2000
FILE_NAME = "corpus.jsonl"

SPLIT_NAME = str  # train/val/test
SPLIT = Dict[SPLIT_NAME, List[TracedTheorem]]
SPLIT_STRATEGY = str

def _split_sequentially(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` sequentially into train/val/test."""
    num_theorems = len(traced_theorems)
    num_train = num_theorems - NUM_VAL - NUM_TEST
    return {
        "train": traced_theorems[:num_train],
        "val": traced_theorems[num_train : num_train + NUM_VAL],
        "test": traced_theorems[num_train + NUM_VAL :],
    }


def split_randomly(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` randomly into train/val/test."""
    logger.info("Splitting the theorems randomly")
    traced_theorems = copy(traced_theorems)
    random.shuffle(traced_theorems)
    return _split_sequentially(traced_theorems)


def split_by_premise(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """
    Split theorems into train/val/test so that proofs in val/test rely on at
    least one novel premise that does not appear in train.
    """
    logger.info("Splitting the theorems by premises")

    # Figure out the number of theorems in train/val/test.
    num_theorems = len(traced_theorems)
    num_val_test = NUM_VAL + NUM_TEST
    num_train = num_theorems - num_val_test
    theorems_val_test = set()

    # Map each premise to a list of theorems using it.
    theorems_by_premises = defaultdict(list)
    for t in traced_theorems:
        for p in t.get_premise_full_names():
            theorems_by_premises[p].append(t)

    # Sort the premises by the number of theorems using them (in ascending order).
    theorems_by_premises = sorted(theorems_by_premises.items(), key=lambda x: len(x[1]))

    # For each premise, put all theorems using it into val_test so that it does not appear in train.
    for _, thms in theorems_by_premises:
        if len(theorems_val_test) < num_val_test:
            theorems_val_test.update(thms)

    # All other theorems go to train.
    theorems_train = [t for t in traced_theorems if t not in theorems_val_test]
    theorems_val_test = list(theorems_val_test)
    random.shuffle(theorems_val_test)

    return {
        "train": theorems_train,
        "val": theorems_val_test[:NUM_VAL],
        "test": theorems_val_test[NUM_VAL:],
    }

def split_data(traced_repo: TracedRepo) -> Dict[SPLIT_STRATEGY, SPLIT]:
    # Skip theorems in the Lean 4 repo itself.
    traced_theorems = [
        thm for thm in traced_repo.get_traced_theorems() if not thm.repo.is_lean4
    ]
    logger.info(f"{len(traced_theorems)} theorems in total")

    return {
        "random": split_randomly(traced_theorems),
        "novel_premises": split_by_premise(traced_theorems),
    }

def export_premises(traced_repo: TracedRepo, dst_path: Path) -> None:
    """Export all premise definitions in a traced repo to ``dst_path``."""
    oup_path = dst_path / FILE_NAME
    if oup_path.exists():
        logger.info(f"{oup_path} already exists. Removing it now.")
        oup_path.unlink()  # Removes the file
    num_premises = 0

    with oup_path.open("wt") as oup:
        G = traced_repo.traced_files_graph

        for tf_node in reversed(list(nx.topological_sort(G))):
            tf = G.nodes[tf_node]["traced_file"]
            imports = [str(_) for _ in G.successors(tf_node)]
            premises = tf.get_premise_definitions()
            num_premises += len(premises)
            oup.write(
                json.dumps(
                    {"path": str(tf.path), "imports": imports, "premises": premises}
                )
                + "\n"
            )
    logger.info(
        f"{num_premises} theorems/definitions from {len(traced_repo.traced_files)} files saved to {oup_path}"
    )

def export_data(
    traced_repo: TracedRepo,
    splits: Dict[SPLIT_STRATEGY, SPLIT],
    dst_path: Union[str, Path],
    **kwargs,
) -> None:
    """Export a traced repo whose theorems have been splitted to ``dst_path``."""
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)
    # if dst_path.exists():
    #     logger.warning(f"{dst_path} already exists. Removing it now.")
    #     shutil.rmtree(dst_path)

    # Export the premises (theorems, definitions, etc.).
    logger.info(f"dst path is {dst_path}")
    export_premises(traced_repo, dst_path)

repo = LeanGitRepo(URL, COMMIT)
traced_repo = trace(repo)
splits = split_data(traced_repo)
export_data(traced_repo, splits, DST_DIR, dataset_name="LeanCopilotBot Corpus")

