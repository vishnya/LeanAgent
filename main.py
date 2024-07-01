"""
This is the driver for LeanCopilotBot. It will do the following things:
1. Search for repositories with Lean files in them.
2. Clone these repositories.
3. Finds theorems with `sorry` in them and replaces them with a proof.
4. Commits and pushes the changes.
5. Opens a pull request.

The goal is to make it easy for people to contribute to the Lean community, and
to generate lots of data for further progress.

Let's make some magic!
"""

import os
import urllib
import requests
import subprocess
import re
import shutil
from dotenv import load_dotenv
from prover import evaluate
from lean_dojo import *
import os
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache
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

from common import set_logger
from prover.proof_search import Status, DistributedProver
import ray
import re
import lean_dojo
from lean_dojo import *
from lean_dojo.constants import LEAN4_PACKAGES_DIR

import pytorch_lightning as pl
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

random.seed(3407)  # https://arxiv.org/abs/2109.08203
_LEAN4_VERSION_REGEX = re.compile(r"leanprover/lean4:(?P<version>.+?)")
repo_dir = "/raid/adarsh/repos" # TODO: for release change these back to <DIR>
DST_DIR = Path("/raid/adarsh/data")
NUM_VAL = NUM_TEST = 2000
FILE_NAME = "corpus.jsonl"
RESULTS_FILE = "results.json"

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

def export_premises(traced_repo: TracedRepo, dst_path: Path, repo_no_dir, sha) -> None:
    """Export all premise definitions in a traced repo to ``dst_path``."""
    oup_path = dst_path / repo_no_dir / sha
    corpus_file = oup_path / FILE_NAME
    if corpus_file.exists():
        logger.info(f"{corpus_file} already exists. Using existing file")
        # TODO: find a better workaround later
        return 0, 0
    oup_path.mkdir(parents=True, exist_ok=True)
    num_premises = 0

    with corpus_file.open("wt") as oup:
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
    return num_premises, len(traced_repo.traced_files)

def export_data(
    traced_repo: TracedRepo,
    splits: Dict[SPLIT_STRATEGY, SPLIT],
    dst_path: Union[str, Path],
    repo_no_dir,
    sha,
    **kwargs,
) -> None:
    """Export a traced repo whose theorems have been splitted to ``dst_path``."""
    logger.info(f"Exporting data to path: {dst_path}/{repo_no_dir}/{sha}")
    return export_premises(traced_repo, dst_path, repo_no_dir, sha)

load_dotenv()

# Feel free to remove any repos from this list if you would like to test on them
known_repositories = [
    "leanprover-community/mathlib4",  # ReProver is trained on this + LeanDojo already tests on it
    "leanprover-community/batteries", # ReProver is trained on this + LeanDojo already tests on it
    "lecopivo/SciLean",
    "leanprover-community/aesop",
    "kmill/lean4-raytracer",
    "AlexKontorovich/PrimeNumberTheoremAnd",
    "teorth/pfr",
    "lurk-lab/yatima",
    "leanprover-community/ProofWidgets4",
    "google-deepmind/debate",
    "leanprover-community/NNG4",
    "teorth/symmetric_project",
    "ufmg-smite/lean-smt",
    "PatrickMassot/GlimpseOfLean",
    "ImperialCollegeLondon/formalising-mathematics-2024",
    "avigad/lamr",
    "leanprover-community/quote4",
    "digama0/lean4lean",
    "leanprover/verso",
    "leanprover-community/iris-lean",
    "avigad/mathematics_in_lean_source",
    "sinhp/Poly",
    "m4lvin/lean4-pdl",
    "leanprover/lean4",
    "leanprover-community/mathlib",
    "leanprover/std4",  # moved to batteries
    "leanprover-community/duper",  # functional programming instead of math
    "leanprover/lake",
    "uwdb/Cosette",
    "AndrasKovacs/smalltt",
    "dselsam/certigrad",
    "Kha/electrolysis",
    "ImperialCollegeLondon/formalising-mathematics",
    "ImperialCollegeLondon/natural_number_game",
    "kbuzzard/xena",
    "leanprover-community/lean4-metaprogramming-book",
    "leanprover-community/tutorials",
    "leanprover-community/lean-liquid",
    "formalabstracts/formalabstracts",
    "ImperialCollegeLondon/M40001_lean",
    "ImperialCollegeLondon/formalising-mathematics-2022",
    "openai/lean-gym",
]

repos = []  # stores the names of all the repos
lean_git_repos = []  # stores the LeanGitRepo objects

personal_access_token = os.environ.get("PERSONAL_ACCESS_TOKEN")

PR_TITLE = "[LeanDojoBot] `sorry` Removed"

PR_BODY = """We identify the files containing theorems that have `sorry`, and replace them with a proof discovered using [LeanDojo](https://github.com/lean-dojo/LeanDojo) and [ReProver](https://github.com/lean-dojo/ReProver).

---

<i>~LeanDojoBot - From the [LeanDojo](https://leandojo.org/) family</i>

[:octocat: repo](https://github.com/Adarsh321123/CS159FinalProject) | [🙋🏾 issues](https://github.com/Adarsh321123/CS159FinalProject/issues)
"""

TMP_BRANCH = "_LeanCopilotBot"

COMMIT_MESSAGE = "[LeanDojoBot] `sorry` Removed"

def clone_repo(repo_url):
    """Clone a git repository and return the path to the repository and its sha."""
    repo_name = "/".join(repo_url.split('/')[-2:]).replace('.git', '')
    print(f"Cloning {repo_url}")
    print(f"Repo name: {repo_name}")
    repo_name = repo_dir + "/" + repo_name
    if os.path.exists(repo_name):
        print(f"Deleting existing repository directory: {repo_name}")
        shutil.rmtree(repo_name)
    subprocess.run(["git", "clone", repo_url, repo_name])
    process = subprocess.Popen(["git", "ls-remote", repo_url], stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    sha = re.split(r'\t+', stdout.decode('utf-8'))[0]
    return repo_name, sha

def branch_exists(repo_name, branch_name):
    """Check if a branch exists in a git repository."""
    proc = subprocess.run(["git", "-C", repo_name, "branch", "-a"], capture_output=True, text=True)
    branches = proc.stdout.split('\n')
    local_branch = branch_name
    remote_branch = f'remote/{branch_name}'
    return any(branch.strip().endswith(local_branch) or branch.strip().endswith(remote_branch) for branch in branches)

def create_or_switch_branch(repo_name, branch_name, base_branch):
    """Create a branch in a git repository if it doesn't exist, or switch to it if it does."""
    if not branch_exists(repo_name, branch_name):
        subprocess.run(["git", "-C", repo_name, "checkout", "-b", branch_name], check=True)
    else:
        subprocess.run(["git", "-C", repo_name, "checkout", branch_name], check=True)
        subprocess.run(["git", "-C", repo_name, "merge", base_branch, "-m", f"Merging {branch_name} into {base_branch}"], check=True)

def commit_changes(repo_name, commit_message):
    """Commit changes to a git repository."""
    status = subprocess.run(["git", "-C", repo_name, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if status == "":
        print("No changes to commit.")
        return False
    subprocess.run(["git", "-C", repo_name, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_name, "commit", "-m", commit_message], check=True)
    return True

def push_changes(repo_name, branch_name):
    """Push changes to a git repository."""
    subprocess.run(["git", "-C", repo_name, "push", "-u", "origin", branch_name], check=True)

def get_default_branch(repo_full_name):
    """Get the default branch of a repository (default `main`)."""
    url = f"https://api.github.com/repos/{repo_full_name}"
    headers = {
        "Authorization": f"token {personal_access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['default_branch']
    else:
        logger.info(f"Failed to get default branch for {repo_full_name}")
        return "main"

def create_pull_request(repo_full_name, title, body, head_branch):
    """Create a pull request in a repository."""
    base_branch = get_default_branch(repo_full_name)
    url = f"https://api.github.com/repos/{repo_full_name}/pulls"
    headers = {
        "Authorization": f"token {personal_access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Pull request created successfully: " + response.json()['html_url'])
        return response.json()['html_url']
    else:
        print("Failed to create pull request", response.text)
        return ""

def search_github_repositories(language="Lean", num_repos=10):
    """Search for the given number of repositories on GitHub that have the given language."""
    headers = {'Authorization': personal_access_token}
    query_params = {
        'q': f'language:{language}',
        'sort': 'stars',
        'order': 'desc',
        'per_page': 100,
    }
    response = requests.get('https://api.github.com/search/repositories', headers=headers, params=query_params)
    
    if response.status_code == 200:
        repositories = response.json()['items']
        cloned_count = 0
        for repo in repositories:
            if cloned_count >= num_repos:
                break
            repo_full_name = repo['full_name']
            if repo_full_name not in known_repositories:
                name = None
                try:
                    clone_url = repo['clone_url']
                    repo_name, sha = clone_repo(clone_url)
                    name = repo_name
                    url = clone_url.replace('.git', '')
                    lean_git_repo = LeanGitRepo(url, sha)
                    lean_git_repos.append(lean_git_repo)
                    repos.append(repo_full_name)
                    cloned_count += 1
                except Exception as e:
                    shutil.rmtree(name)
                    # Note: some 404s happen since some repos don't have a lean-toolchain
                    # but still use Lean
                    print(f"Failed to clone {repo_full_name} because of {e}")
    else:
        print("Failed to search GitHub", response.status_code)

def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    return m["version"]

def is_supported_version(v) -> bool:
    """
    Check if ``v`` is at least `v4.3.0-rc2` and at most `v4.8.0-rc2`.
    Note: Lean versions are generally not backwards-compatible. Also, the Lean FRO
    keeps bumping the default versions of repos to the latest version, which is
    not necessarily the latest stable version. So, we need to be careful about
    what we choose to support.
    """
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3) or (major == 4 and minor > 8) or (major == 4 and minor == 8 and patch > 0):
        return False
    if (
        major > 4
        or (major == 4 and minor > 3)
        or (major == 4 and minor == 3 and patch > 0)
    ):
        return True
    assert major == 4 and minor == 3 and patch == 0
    if "4.3.0-rc" in v:
        rc = int(v.split("-")[1][2:])
        return rc >= 2
    elif "4.8.0-rc" in v:
        rc = int(v.split("-")[1][2:])
        return rc <= 2
    else:
        return True

def update_and_train(model_checkpoint_path, new_data_path, max_epochs=10):
    # Load the model from checkpoint
    if not torch.cuda.is_available():
        logger.warning("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # TODO: reduce repetition in code like this
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "max_seq_len": 512,
        "num_retrieved": 100,
    }

    # state_dict = torch.load(model_checkpoint_path)
    # state_dict["pytorch-lightning_version"] = "0.0.0"
    # state_dict['global_step'] = None
    # state_dict['epoch'] = None
    # state_dict['state_dict'] = state_dict
    # state_dict['callbacks'] = None
    # state_dict['loops'] = {}
    # state_dict['legacy_pytorch-lightning_version'] = None
    # torch.save(state_dict, model_checkpoint_path)
    
    model = PremiseRetriever.load(
        model_checkpoint_path, device, freeze=False, config=config
    )

    # Assuming your data module can handle updating, update data source
    # TODO: try with novel split later instead of random
    # TODO: use the yaml file instead of repeating here, same throughout
    corpus_path = new_data_path.split("/")[0] + "/corpus.jsonl"
    data_module = RetrievalDataModule(
        data_path=new_data_path,
        corpus_path=corpus_path,
        num_negatives=3,
        num_in_file_negatives=1,
        model_name="google/byt5-small",
        batch_size=8,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=4
    )
    data_module.setup(stage='fit')

    # checkpoint_callback = ModelCheckpoint(
    #     save_weights_only=False,  # This needs to be False to save full state
    #     verbose=True,
    #     save_top_k=1,
    #     monitor='Recall@10_val',
    #     mode='max',
    # )

    # TODO: save checkpoint after every epoch end? or is that already done?
    # trainer = pl.Trainer(
    #     max_epochs=max_epochs,
    #     callbacks=[checkpoint_callback],
    # )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
    )

    # Continue training
    logger.info("Continuing training...")

    import ipdb; ipdb.set_trace()
    trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)

    import ipdb; ipdb.set_trace()

    # TODO: save checkpoint if the loss is good
    # TODO: for a given repo, run the benchmark code (check for differences with courpus code here) to get the data nad corpus
    # TODO: we now want to add these new things to the exisitng dataset for ONLY the current repo like PFR, not its dependencies, so delete hte lines for existing train/val/test and corpus with current repo and add the new ones
    # TODO: then run this online learning code
    
    # TODO: should we use retriever trained on novel premises or random?

    # TODO: use kaiyuy/leandojo-pl-ckpts for generator too if it works?

    # TODO: delete new folder after done merging, call folders like existing and new or something and move to raid

    return model

def retrieve_proof(repo, repo_no_dir, sha):
    """
    This method does the following:
    1. Check if the given repo is supported.
    2. Trace the repo.
    3. Generate a corpus of the repo's premises.
    4. Search for proofs for theorems with `sorry` in them.
    """
    ckpt_path = "/raid/adarsh/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_lightning.ckpt"
    indexed_corpus_path = str(DST_DIR / repo_no_dir / sha) + "/corpus.jsonl"
    tactic = None
    module = None
    num_workers = 5
    num_gpus = 1
    timeout = 600
    num_sampled_tactics = 64
    verbose = False

    # we need to change the toolchain version that the bot uses
    # to match the repo we are currently tracing
    config = repo.get_config("lean-toolchain")
    logger.info(f"lean toolchain version: {config}")
    v = get_lean4_version_from_config(config["content"])
    logger.info(f"lean version v: {v}")
    logger.info(f"is supported: {is_supported_version(v)}")
    if not is_supported_version(v):
        logger.info("Unsupported version")
        return None
    v = v[1:] # ignore "v" at beginning
    lean_dir = "/home/adarsh/.elan/toolchains/leanprover--lean4---" + v
    logger.info(f"lean path {lean_dir}")
    if not os.path.exists(lean_dir):
        logger.info(f"Lean toolchain path does not exist: {lean_dir}")
        return None
    os.environ['LEAN4_PATH'] = lean_dir
    os.environ['PATH'] = f"{lean_dir}/bin:{os.environ.get('PATH', '')}"
    logger.info(f"Switched to Lean toolchain at: {lean_dir}")
    logger.info(f"lean --version: {subprocess.run(['lean', '--version'], capture_output=True).stdout.decode('utf-8')}")
    logger.info(f"repo: {repo}")

    if ray.is_initialized():
        ray.shutdown()

    traced_repo = None
    try:
        traced_repo = trace(repo)
    except Exception as e:
        logger.info(f"Failed to trace repo {repo} because of {e}")
        return None

    # TODO: don't resplit if the corpus already exists
    logger.info("MAIN: about to split data")
    splits = split_data(traced_repo)
    logger.info("MAIN: done splitting data")
    logger.info("MAIN: about to export corpus.jsonl")
    num_premises, num_files_traced = export_data(traced_repo, splits, DST_DIR, repo_no_dir, sha, dataset_name="LeanCopilotBot Corpus")
    logger.info("MAIN: exported corpus.jsonl")

    data = []

    thms = traced_repo.get_traced_theorems()
    for thm in thms:
        if not thm.has_tactic_proof():
            continue
        proof = thm.get_tactic_proof()
        if proof is None:
            continue
        theorem = thm.theorem
        start = thm.start
        end = thm.end
        data.append((theorem, proof, start, end))

    # TODO: optimize
    logger.info(len(data))
    theorems = []
    positions = []
    ends = []
    for elem in data:
        cur_repo = elem[0].repo
        cur_theorem = elem[0]
        cur_proof = elem[1]
        if (cur_repo == repo) and ("sorry" in cur_proof):  # avoid proving theorems in Lean4 and other dependencies
            theorems.append(cur_theorem)
            positions.append(elem[2])
            ends.append(elem[3])

    num_sorries = len(theorems)
    logger.info(f"Found {num_sorries} sorries!")
    logger.info("MAIN: about to search for proofs")
    prover = DistributedProver(
        ckpt_path,
        indexed_corpus_path,
        tactic,
        module,
        num_workers,
        num_gpus=num_gpus,
        timeout=timeout,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
    )
    results = prover.search_unordered(repo, theorems, positions)
    logger.info("MAIN: done searching for proofs")
    proofs = []
    unproved_sorries = []
    for result in results:
        if result is not None:
            if result.status == Status.PROVED:
                proof_text = "\n".join(result.proof)
                # TODO: find more efficient way to get url and repo name
                repo_name = "/".join(result.theorem.repo.url.split('/')[-2:]).replace('.git', '')
                repo_name = repo_dir + "/" + repo_name
                file_path = repo_name + "/" + str(result.theorem.file_path)
                theorem_name = str(result.theorem.full_name)
                start = None
                end = None
                # TODO: optimize
                for i in range(len(theorems)):
                    if theorems[i] == result.theorem:
                        start = positions[i]
                        end = ends[i]
                proofs.append((file_path, start, end, proof_text, theorem_name))
            else:
                repo_name = "/".join(result.theorem.repo.url.split('/')[-2:]).replace('.git', '')
                repo_name = repo_dir + "/" + repo_name
                file_path = repo_name + "/" + str(result.theorem.file_path)
                theorem_name = str(result.theorem.full_name)
                unproved_sorries.append((file_path, theorem_name))

    return num_sorries, proofs, num_premises, num_files_traced, unproved_sorries

def replace_sorry_with_proof(proofs):
    """Replace the `sorry` with the proof text in the Lean files."""
    logger.info(f"Replacing sorries with {len(proofs)} proofs!")
    # Group proofs by file paths
    proofs_by_file = {}
    for proof in proofs:
        file_path, start, end, proof_text, theorem_name = proof
        if file_path not in proofs_by_file:
            proofs_by_file[file_path] = []
        proofs_by_file[file_path].append((start, end, proof_text))
    
    for file_path, proofs in proofs_by_file.items():
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # sort proof by starting line and column number (working bottom up retains positions)
        proofs.sort(key=lambda x: (x[0].line_nb, x[0].column_nb), reverse=True)
        
        for start, end, proof_text in proofs:
            start_line, start_col = start.line_nb - 1, start.column_nb - 1
            end_line, end_col = end.line_nb - 1, end.column_nb - 1
            original_text = ''.join(lines[start_line:end_line + 1])
            new_text = original_text.replace('sorry', proof_text, 1)
            lines[start_line:end_line + 1] = new_text
            
            with open(file_path, 'w') as file:
                file.writelines(lines)


def main():
    """The main function that drives the bot."""
    # Update the encoder weights using an updated corpus
    # TODO: move data to raid
    # TODO: make sure this checkpoint works without online
    # TODO: change kaiyuy files to robert checkpoint everywhere
    # model_checkpoint_path = "checkpoints/new_retriever.ckpt"
    model_checkpoint_path = "leandojo-pl-ckpts/new_retriever.ckpt"
    # model_checkpoint_path = "/raid/adarsh/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_lightning.ckpt"
    new_checkpoint = update_and_train(model_checkpoint_path, "new_version_test_benchmark/random")

    results = {
        "total_repositories": 0,
        "repositories": {}
    }
    # TODO: remove
    clone_url = "https://github.com/Adarsh321123/new-version-test.git"
    repo_name, sha = clone_repo(clone_url)
    name = repo_name
    url = clone_url.replace('.git', '')
    lean_git_repo = LeanGitRepo(url, sha)
    lean_git_repos.append(lean_git_repo)
    repos.append("Adarsh321123/new-version-test")
    # search_github_repositories("Lean", 10)
    # repos.append("Adarsh321123/pfr")
    # lean_git_repos.append(LeanGitRepo("https://github.com/Adarsh321123/pfr", "c297615a183a32f29f1313895c32f72d200f0bb6"))
    num_repos = len(repos)
    results["total_repositories"] = num_repos
    print(f"Found {num_repos} repositories")
    for i in range(num_repos):
        repo = repos[i]
        repo_no_dir = repo
        repo = repo_dir + "/" + repo
        lean_git_repo = lean_git_repos[i]
        print(f"Processing {repo}")
        results["repositories"][repo] = {
            "number_of_sorries": 0,
            "number_of_proofs_found": 0,
            "proofs_details": [],
            "unproved_sorries": [],
            "number_of_premises_theorems_retrieved": 0,
            "num_files_traced": 0,
            "PR": "",
        }
        base_branch = get_default_branch(repo_no_dir)
        subprocess.run(["git", "-C", repo, "fetch", "origin", base_branch], check=True)
        subprocess.run(["git", "-C", repo, "checkout", base_branch], check=True)
        subprocess.run(["git", "-C", repo, "pull", "origin", base_branch], check=True)
        create_or_switch_branch(repo, TMP_BRANCH, base_branch)
        result = retrieve_proof(lean_git_repo, repo_no_dir, lean_git_repo.commit)
        if result is None:
            logger.info("Skipping repository due to configuration or error.")
            continue
        num_sorries, proofs, num_premises, num_files_traced, unproved_sorries = result
        if proofs is None:
            continue
        results["repositories"][repo]["number_of_sorries"] = num_sorries
        results["repositories"][repo]["number_of_proofs_found"] = len(proofs)
        results["repositories"][repo]["number_of_premises_theorems_retrieved"] = num_premises
        results["repositories"][repo]["num_files_traced"] = num_files_traced
        for proof in proofs:
            results["repositories"][repo]["proofs_details"].append({
                "file_path": proof[0],
                "theorem_name": proof[4],
                "proof_text": proof[3]
            })
        for unproved_sorry in unproved_sorries:
            results["repositories"][repo]["unproved_sorries"].append({
                "file_path": unproved_sorry[0],
                "theorem_name": unproved_sorry[1]
            })
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        # Uncomment if you would like to contribute back to the repos!
        # replace_sorry_with_proof(proofs)
        # committed = commit_changes(repo, COMMIT_MESSAGE)
        # if committed:
        #     push_changes(repo, TMP_BRANCH)
        #     url = str(create_pull_request(repo_no_dir, PR_TITLE, PR_BODY, TMP_BRANCH))
        #     results["repositories"][repo]["PR"] = url
        # shutil.rmtree(repo)

if __name__ == "__main__":
    main()
