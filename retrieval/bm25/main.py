"""Script for training the BM25 premise retriever."""

import os
import ray
import json
import pickle
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing
from loguru import logger
from common import Corpus
from lean_dojo import Pos
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer
from typing import List, Dict, Any
from ray.util.actor_pool import ActorPool


from common import Context, format_state, get_all_pos_premises


def _process_theorem(
    thm: Dict[str, Any],
    corpus: Corpus,
    tokenizer,
    bm25,
    num_retrieved: int,
    use_all_premises: bool,
) -> List[Dict[str, Any]]:
    logger.info(f"Processing {thm['full_name']} at {thm['file_path']}")
    preds = []
    file_path = thm["file_path"]

    if use_all_premises:
        accessible_premise_idxs = list(range(len(corpus)))
    else:
        accessible_premise_idxs = corpus.get_accessible_premise_indexes(
            file_path, Pos(*thm["start"])
        )

    for i, tac in enumerate(thm["traced_tactics"]):
        state = format_state(tac["state_before"])
        ctx = Context(file_path, thm["full_name"], Pos(*thm["start"]), state)
        tokenized_ctx = tokenizer.encode(ctx.serialize()).tokens

        scores = np.array(bm25.get_batch_scores(tokenized_ctx, accessible_premise_idxs))
        scores_idxs = np.argsort(scores)[::-1][:num_retrieved]
        retrieved_idxs = [accessible_premise_idxs[i] for i in scores_idxs]
        retrieved_premises = [corpus[i] for i in retrieved_idxs]
        retrieved_scores = scores[scores_idxs].tolist()

        all_pos_premises = get_all_pos_premises(tac["annotated_tactic"], corpus)
        preds.append(
            {
                "url": thm["url"],
                "commit": thm["commit"],
                "file_path": thm["file_path"],
                "full_name": thm["full_name"],
                "start": thm["start"],
                "tactic_idx": i,
                "context": ctx,
                "all_pos_premises": all_pos_premises,
                "retrieved_premises": retrieved_premises,
                "scores": retrieved_scores,
            }
        )

    logger.info(f"Processed {thm['full_name']} at {thm['file_path']}")
    return preds


@ray.remote(num_cpus=1)
class TheoremProcessor:
    def __init__(
        self,
        tokenizer_path: str,
        data_path: str,
        num_retrieved: int,
        use_all_premises: bool,
    ) -> None:
        logger.info("Initializing theorem processor")
        self.num_retrieved = num_retrieved
        self.use_all_premises = use_all_premises

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.corpus = Corpus(os.path.join(data_path, "../corpus.jsonl"))
        premises = [premise.serialize() for premise in self.corpus.all_premises]
        tokenized_premises = [self.tokenizer.encode(p).tokens for p in premises]
        self.bm25 = BM25Okapi(tokenized_premises)
        logger.info("Finished initializing theorem processor")

    def process_theorem(self, thm: Dict[str, Any]):
        try:
            return _process_theorem(
                thm,
                self.corpus,
                self.tokenizer,
                self.bm25,
                self.num_retrieved,
                self.use_all_premises,
            )
        except Exception as e:
            logger.error(f"Error processing theorem {thm['full_name']}: {str(e)}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for training the BM25 premise retriever."
    )
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument("--num-retrieved", type=int, default=100)
    parser.add_argument("--use-all-premises", action="store_true")
    parser.add_argument("--num-cpus", type=int, default=32)
    args = parser.parse_args()
    logger.info(args)

    if multiprocessing.cpu_count() < args.num_cpus:
        logger.warning(
            f"Number of cpus requested ({args.num_cpus}) is greater than the number of cpus available ({multiprocessing.cpu_count()})"
        )

    logger.info("About to load theorems")
    with open(os.path.join(args.data_path, "test.json"), 'r') as file:
        theorems = json.load(file)
    logger.info("Finished loading theorems")

    if args.num_cpus > 1:
        logger.info("About to create a pool")
        pool = ActorPool(
            [
                TheoremProcessor.remote(
                    args.tokenizer_path,
                    args.data_path,
                    args.num_retrieved,
                    args.use_all_premises,
                )
                for _ in range(args.num_cpus)
            ]
        )
        logger.info("Finished making a pool")
        logger.info("About to process theorems")
        futures = pool.map_unordered(lambda actor, thm: actor.process_theorem.remote(thm), theorems)
        preds = [ray.get(f) for f in tqdm(futures, total=len(theorems))]
        logger.info("Finished processing theorems")
    else:
        logger.info("Initializing tokenizer")
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        logger.info("Finished initializing tokenizer")
        logger.info("Initializing corpus")
        corpus = Corpus(os.path.join(args.data_path, "../corpus.jsonl"))
        logger.info("Finished initializing corpus")
        logger.info("Initializing premises")
        premises = [premise.serialize() for premise in corpus.all_premises]
        logger.info("Finished initializing premises")
        logger.info("Initializing tokenized premises")
        tokenized_premises = [tokenizer.encode(p).tokens for p in premises]
        logger.info("Finished initializing tokenized premises")
        logger.info("Initializing BM25")
        bm25 = BM25Okapi(tokenized_premises)
        logger.info("Finished initializing BM25")
        logger.info("About to process theorems")
        preds = list(
            itertools.chain.from_iterable(
                [
                    _process_theorem(
                        thm,
                        corpus,
                        tokenizer,
                        bm25,
                        args.num_retrieved,
                        args.use_all_premises,
                    )
                    for thm in tqdm(theorems)
                ]
            )
        )

        logger.info("Finished processing theorems")

    logger.info("About to save predictions")
    with open(args.output_path, "wb") as oup:
        pickle.dump(preds, oup)
    logger.info(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
