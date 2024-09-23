# leanagent.py

import warnings
import logging
import os
import torch
from loguru import logger
from lean_dojo import LeanGitRepo, Theorem, Pos
from retrieval.model import PremiseRetriever
from generator.model_demo import RetrievalAugmentedGenerator
from prover.proof_search_demo import DistributedProver, SearchResult, Status

ROOT_DIR = os.environ.get('ROOT_DIR', '/data/yingzi_ma/lean_project')
DATA_DIR = os.path.join(ROOT_DIR, "datasets_demo")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints_demo")
CORPUS_EMBEDDINGS_CACHE = os.path.join(DATA_DIR, "corpus_embeddings_cache.pt")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Lightning automatically upgraded your loaded checkpoint")
warnings.filterwarnings("ignore", message="Found keys that are in the model state dict but not in the checkpoint")
warnings.filterwarnings("ignore", message="Found keys that are not in the model state dict but in the checkpoint")

def load_model(ret_checkpoint_path, device):
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
        "ret_ckpt_path": ret_checkpoint_path,
    }
    return RetrievalAugmentedGenerator.load(f"{ROOT_DIR}/checkpoint_tacgen.ckpt", device=device, freeze=True, config=config)

class LeanAgent:
    def __init__(self, use_baseline=False):
        if use_baseline:
            print("Initializing baseline...")
        else:
            print("Initializing LeanAgent...")
        self.use_baseline = use_baseline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading the model...")
        self.ret_checkpoint_path = os.path.join(CHECKPOINT_DIR, "baseline_checkpoint.ckpt" if use_baseline else "leanagent_checkpoint.ckpt")
        self.model = load_model(self.ret_checkpoint_path, self.device)
        print("Model loaded!")
        self.setup_prover()
        if use_baseline:
            print("Baseline is ready!")
        else:
            print("LeanAgent is ready!")

    def setup_prover(self):
        use_vllm = False
        corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
        self.model.retriever.load_corpus(corpus_path)

        if os.path.exists(CORPUS_EMBEDDINGS_CACHE):
            embeddings = torch.load(CORPUS_EMBEDDINGS_CACHE, map_location=self.device)
            self.model.retriever.corpus_embeddings = embeddings.to(self.device)
            self.model.retriever.embeddings_staled = False
        else:
            self.model.retriever.reindex_corpus(batch_size=32)
            torch.save(self.model.retriever.corpus_embeddings, CORPUS_EMBEDDINGS_CACHE)

        num_workers = 1
        num_gpus = 1 if torch.cuda.is_available() else 0
        timeout = 10
        max_expansions = None
        num_sampled_tactics = 64
        debug = False

        self.prover = DistributedProver(
            use_vllm,
            self.model,
            corpus_path,
            None,
            None,
            num_workers,
            num_gpus,
            timeout,
            max_expansions,
            num_sampled_tactics,
            debug,
        )

    def prove_theorem(self, repo_url: str, file_path: str, theorem_name: str) -> SearchResult:
        repo_commit, pos = self.get_repo_info(repo_url)
        repo = LeanGitRepo(repo_url, repo_commit)
        theorem = Theorem(repo, file_path, theorem_name)
        print(f"Proving {theorem_name}")
        results = self.prover.search_unordered(repo, [theorem], [pos])
        return results[0] if results else None
    
    @staticmethod
    def get_repo_info(repo_url):
        # TODO: chagen pos based on theorme
        if "miniF2F" in repo_url:
            return "9e445f5435407f014b88b44a98436d50dd7abd00", Pos(517, 1)
        elif "mathematics_in_lean_source" in repo_url:
            return "5297e0fb051367c48c0a084411853a576389ecf5", Pos(170, 1)
        return None, None

def format_proof(proof):
    if not proof:
        return "No proof found."
    
    formatted_proof = "Proof steps:\n"
    for i, step in enumerate(proof, 1):
        formatted_proof += f"{i}. {step}\n"
    return formatted_proof

def print_proof_result(result):
    print(f"\nTheorem: {result.theorem.full_name}")
    print(f"Status: {'Proved' if result.status == Status.PROVED else 'Not Proved'}")
    print(f"\n{format_proof(result.proof)}")

def main():
    # super easy setup: pip install git+https://github.com/Adarsh321123/leanagent.git
    # python leanagent.py

    print("\nDemo: Continuously Improving")
    agent = LeanAgent()
    result = agent.prove_theorem(
        repo_url="https://github.com/yangky11/miniF2F-lean4",
        file_path="MiniF2F/Test.lean",
        theorem_name="mathd_algebra_160"
    )
    print_proof_result(result)

    # baseline_agent = LeanAgent(use_baseline=True)
    # baseline_result = agent.prove_theorem(
    #     repo_url="https://github.com/yangky11/miniF2F-lean4",
    #     file_path="MiniF2F/Test.lean",
    #     theorem_name="induction_12dvd4expnp1p20"
    # )
    # print_proof_result(result)
    # # TODO: update
    # print(f"Baseline model succeeded: {baseline_result.status == Status.PROVED if baseline_result else False}")
    # print("LeanAgent succeeded, demonstrating significant performance improvement")

    # print("\nDemo: Continuously Generalizable")
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