# leanagent.py

import os
import torch
from loguru import logger
from lean_dojo import LeanGitRepo, Theorem, Pos
from retrieval.model import PremiseRetriever
from generator.model import RetrievalAugmentedGenerator
from prover.proof_search_demo import DistributedProver, SearchResult

# TODO: update
RAID_DIR = os.environ.get('RAID_DIR', '/data/yingzi_ma/lean_project')
DATA_DIR = os.path.join(RAID_DIR, "datasets_demo")
CHECKPOINT_DIR = os.path.join(RAID_DIR, "checkpoints_demo")

class LeanAgent:
    def __init__(self, use_baseline=False):
        print("Initializing LeanAgent...")
        self.use_baseline = use_baseline
        # TODO: try with cpu nad gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()
        self.setup_prover()
        self.checkpoint_path = None
        print("LeanAgent is ready!")

    def load_models(self):
        # TODO: see if we can use lean4lean checkpoint for both demos
        if self.use_baseline:
            self.checkpoint_path = os.path.join(CHECKPOINT_DIR, "baseline_checkpoint.ckpt")
        else:
            self.checkpoint_path = os.path.join(CHECKPOINT_DIR, "leanagent_checkpoint.ckpt")

        config = {
            "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
            "lr": 1e-3,
            "warmup_steps": 1000,
            "num_beams": 5,
            "eval_num_retrieved": 10,
            "eval_num_workers": 1,
            "eval_num_gpus": 1,
            "eval_num_theorems": 100,
            "max_inp_seq_len": 512,
            "max_oup_seq_len": 128,
            "ret_ckpt_path": self.checkpoint_path,
        }

        self.tac_gen = RetrievalAugmentedGenerator.load(
            self.checkpoint_path, device=self.device, freeze=True, config=config
        )
        logger.info(f"Loaded model from {self.checkpoint_path}")

    def setup_prover(self):
        use_vllm = False
        # TODO: make it specific to dataset
        corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
        num_workers = 1
        num_gpus = 1 if torch.cuda.is_available() else 0
        timeout = 60
        max_expansions = None
        num_sampled_tactics = 64
        debug = False
        ckpt_path = f"{RAID_DIR}/leanagent_checkpoint_tacgen.ckpt"

        self.prover = DistributedProver(
            use_vllm,
            ckpt_path,
            corpus_path,
            None,
            None,
            num_workers,
            num_gpus,
            timeout,
            max_expansions,
            num_sampled_tactics,
            self.checkpoint_path,
            debug,
        )

    def prove_theorem(self, repo_url: str, file_path: str, theorem_name: str) -> SearchResult:
        repo_commit = None
        pos = None
        if "miniF2F" in repo_url:
            repo_commit = "9e445f5435407f014b88b44a98436d50dd7abd00"
            pos = Pos(517, 1)
        elif "mathematics_in_lean_source" in repo.url:
            repo_commit = "5297e0fb051367c48c0a084411853a576389ecf5"
            pos = Pos(170, 1)
        repo = LeanGitRepo(repo_url, repo_commit)
        theorem = Theorem(repo, file_path, theorem_name)
        results = self.prover.search_unordered(repo, [theorem], [pos])
        return results[0] if results else None

def main():
    # super easy setup: pip install git+https://github.com/Adarsh321123/leanagent.git
    # python leanagent.py

    print("\nMiniF2F Demo")
    agent = LeanAgent()
    result = agent.prove_theorem(
        repo_url="https://github.com/yangky11/miniF2F-lean4",
        file_path="MiniF2F/Test.lean",
        theorem_name="induction_12dvd4expnp1p20"
    )
    print(f"Proof found: {result.status == Status.PROVED if result else False}")
    if result and result.status == Status.PROVED:
        print(result.proof)

    # print("\nContinuously Improving")
    # baseline_agent = LeanAgent(use_baseline=True)
    # baseline_result = agent.prove_theorem(
    #     repo_url="https://github.com/yangky11/miniF2F-lean4",
    #     file_path="MiniF2F/Test.lean",
    #     theorem_name="induction_12dvd4expnp1p20"
    # )
    # print(f"Baseline model succeeded: {baseline_result.status == Status.PROVED if baseline_result else False}")
    # print("LeanAgent succeeded, demonstrating significant performance improvement")

    # print("\nContinuously Generalizable")
    # result = agent.prove_theorem(
    #     repo_url="https://github.com/leanprover-community/mathematics_in_lean",
    #     file_path="MIL/C03_Logic/S01_Implication_and_the_Universal_Quantifier.lean",
    #     theorem_name="C03S01.Subset.trans"
    # )
    # print(f"Proof found: {result.status == Status.PROVED if result else False}")
    # if result and result.status == Status.PROVED:
    #     print(result.proof)
    # print("LeanAgent generalizes to new repositories without retraining")

if __name__ == "__main__":
    main()