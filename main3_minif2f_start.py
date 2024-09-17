"""
This is the driver for LeanBot. It will do the following things:
1. Search for repositories with Lean files in them.
2. Clone these repositories.
3. Find theorems with `sorry` in them and replace them with a proof.
4. Commit and push the changes.
5. Open a pull request.
"""

import math
import ray
from collections import defaultdict
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
import numpy as np
import hashlib
import argparse
from tqdm import tqdm
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Pos, is_available_in_cache
from lean_dojo import Theorem as LeanDojoTheorem
import json
import shutil
import random
import networkx as nx
from copy import copy
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Union
import generate_benchmark_lean4
import traceback
import sys
from tqdm import tqdm
from dynamic_database import *
import time
from packaging import version
import psutil
import atexit
from pytorch_lightning.strategies import DDPStrategy

from common import set_logger
from prover.proof_search import Status, DistributedProver, SearchResult
import re
import lean_dojo
from lean_dojo import *
from lean_dojo.constants import LEAN4_PACKAGES_DIR

import pytorch_lightning as pl
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule
from retrieval.main import run_cli
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning import seed_everything

# TODO: standardize with all Path or just string
random.seed(3407)  # https://arxiv.org/abs/2109.08203
# TODO: constant?
# TODO: do we still need repo_dir
BATCH_SIZE=4
RAID_DIR = os.environ.get('RAID_DIR')
repo_dir = f"{RAID_DIR}/repos_new" # TODO: for release change these back to <DIR>

# DATA_DIR = "datasets_PT_merge_all_no_ewc"
# CHECKPOINT_DIR = "checkpoints_PT_merge_all_no_ewc"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_PT_merge_all_no_ewc.txt"
# DB_FILE_NAME = "dynamic_database_PT_merge_all_no_ewc.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_PT_merge_all_no_ewc.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_PT_merge_all_no_ewc.pkl"

# DATA_DIR = "datasets_PT_single_repo_no_ewc"
# CHECKPOINT_DIR = "checkpoints_PT_single_repo_no_ewc"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_PT_single_repo_no_ewc.txt"
# DB_FILE_NAME = "dynamic_database_PT_single_repo_no_ewc.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_PT_single_repo_no_ewc.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_PT_single_repo_no_ewc.pkl"

DATA_DIR = "datasets_PT_single_repo_no_ewc_curriculum"
CHECKPOINT_DIR = "checkpoints_PT_single_repo_no_ewc_curriculum_start"
EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_PT_single_repo_no_ewc_curriculum.txt"
DB_FILE_NAME = "dynamic_database_PT_single_repo_no_ewc_curriculum_full_start.json"
PROOF_LOG_FILE_NAME = "proof_logs/proof_log_PT_single_repo_no_ewc_curriculum_minif2f_start.log"
ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_PT_single_repo_no_ewc_curriculum_full_minif2f.pkl"

# DATA_DIR = "datasets_PT_single_repo_ewc"
# CHECKPOINT_DIR = "checkpoints_PT_single_repo_ewc"
# FISHER_DIR = "fisher_PT_single_repo_ewc"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_PT_single_repo_ewc.txt"
# DB_FILE_NAME = "dynamic_database_PT_single_repo_ewc.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_PT_single_repo_ewc.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_PT_single_repo_ewc.pkl"

# DATA_DIR = "datasets_retrieval_single_repo_no_ewc"
# CHECKPOINT_DIR = "checkpoints_retrieval_single_repo_no_ewc"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_retrieval_single_repo_no_ewc.txt"
# DB_FILE_NAME = "dynamic_database_retrieval_single_repo_no_ewc.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_retrieval_single_repo_no_ewc.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_retrieval_single_repo_no_ewc.pkl"

# DATA_DIR = "datasets_PT_single_repo_ewc_pfr"
# CHECKPOINT_DIR = "checkpoints_PT_single_repo_ewc_pfr"
# FISHER_DIR = "fisher_PT_single_repo_ewc_pfr"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_PT_single_repo_ewc_pfr.txt"
# DB_FILE_NAME = "dynamic_database_PT_single_repo_ewc_pfr.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_PT_single_repo_ewc_pfr.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_PT_single_repo_ewc_pfr.pkl"

# DATA_DIR = "datasets_PT_single_repo_no_ewc_pfr"
# CHECKPOINT_DIR = "checkpoints_PT_single_repo_no_ewc_pfr"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_PT_single_repo_no_ewc_pfr.txt"
# DB_FILE_NAME = "dynamic_database_PT_single_repo_no_ewc_pfr.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_PT_single_repo_no_ewc_pfr.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_PT_single_repo_no_ewc_pfr.pkl"

# DATA_DIR = "datasets_retrieval_single_repo_no_ewc_pfr"
# CHECKPOINT_DIR = "checkpoints_retrieval_single_repo_no_ewc_pfr"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_retrieval_single_repo_no_ewc_pfr.txt"
# DB_FILE_NAME = "dynamic_database_retrieval_single_repo_no_ewc_pfr.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_retrieval_single_repo_no_ewc_pfr.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_retrieval_single_repo_no_ewc_pfr.pkl"

# DATA_DIR = "datasets_retrieval_merge_all_no_ewc"
# CHECKPOINT_DIR = "checkpoints_retrieval_merge_all_no_ewc"
# EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/ReProver/total_evaluation_results_retrieval_merge_all_no_ewc.txt"
# DB_FILE_NAME = "dynamic_database_retrieval_merge_all_no_ewc.json"
# PROOF_LOG_FILE_NAME = "proof_logs/proof_log_retrieval_merge_all_no_ewc.log"
# ENCOUNTERED_THEOREMS_FILE = "encountered_theorems_retrieval_merge_all_no_ewc.pkl"

# TODO: do we still need this?
load_dotenv()

repos_for_merged_dataset = []

# TODO: automate this
# Feel free to remove any repos from this list if you would like to test on them
known_repositories = [
    "leanprover-community/mathlib4",  # ReProver is trained on this
    "leanprover-community/batteries", # functional programming instead of math
    "leanprover-community/aesop",
    "leanprover/lean4",
    "leanprover-community/mathlib", # Mathlib3 version
    "leanprover-community/mathlib3",
    "leanprover/std4",  # moved to batteries
    "leanprover-community/duper",  # functional programming instead of math
    "leanprover/lake",
    "openai/lean-gym",
    # already tested:
    "lecopivo/SciLean",
    "avigad/mathematics_in_lean_source",
    "teorth/pfr",
    "dwrensha/compfiles",
    "digama0/lean4lean",
    "AlexKontorovich/PrimeNumberTheoremAnd",
    # newly tested:
    "leanprover-community/lean4-metaprogramming-book",
    "ImperialCollegeLondon/FLT",
    "kmill/lean4-raytracer",  # no theorems
    "argumentcomputer/yatima",  # trace problems
    "ImperialCollegeLondon/formalising-mathematics-2024",  # trace problems
    "leanprover-community/ProofWidgets4",  # trace problems
    "leanprover/verso",  # trace problems
    "leanprover-community/NNG4",  # trace problems
    "ufmg-smite/lean-smt",  # fails to trace due to windows-style line endings
    "google-deepmind/debate",
    "teorth/symmetric_project",  # no compatible commit
    "cmu-l3/llmlean",  # irrelevant + only 4 theorems
    "PatrickMassot/GlimpseOfLean",   # strange trace problems with _parse_deps
    "avigad/lamr",  # trace problems
    "leanprover-community/quote4",  # no theorems
    "yuma-mizuno/lean-math-workshop",
    "leanprover-community/iris-lean",  # trace problems
    "aripiprazole/rinha",  # incompatible commit
    "loganrjmurphy/LeanEuclid",
    "leanprover/lean4-cli",  # no theorems
    "leanprover/LeanInk",  # no theorems
    "leanprover-community/lean-auto",
    "leanprover-community/repl",  # no theorems
    "leanprover/doc-gen4",  # no theorems
    "leanprover-community/con-nf",
    "FormalizedFormalLogic/Foundation",
    "leanprover/SampCert",  # trace problems
    "nomeata/loogle",
    "risc0/risc0-lean4",
    "siddhartha-gadgil/Saturn",
    # "leanprover/LNSym",
    # "leanprover-community/flt-regular",
    "eric-wieser/lean-matrix-cookbook",
    "PatrickMassot/verbose-lean4",  # no theorems
    "tydeu/lean4-alloy",  # no theorems
    # "opencompl/lean-mlir-old",
    "leanprover/leansat", # deprecated
    "BoltonBailey/formal-snarks-project", # two theorems
    "dwrensha/lean4-maze", # two theorems
    "leanprover-community/mathport", # irrelevant
    # "rami3l/plfl",
    # "HEPLean/HepLean",
    # "forked-from-1kasper/ground_zero",
    # "mo271/formal_book",
    "argumentcomputer/LSpec",  # one theorem
    "reaslab/jixia", # no theorems
    "riccardobrasca/flt3", # no theorems
    "dwrensha/animate-lean-proofs", # irrelevant
    "lean-ja/lean-by-example", # irrelevant
    "NethermindEth/Clear", # no theorems
    "fgdorais/lean4-parser", # irrelevant
    "semorrison/lean-training-data", # irrelevant
    "verse-lab/lean-ssr", # irrelevant
    "GaloisInc/lean-llvm", # irrelevant
    "argumentcomputer/Wasm.lean", # irrelevant
    "NethermindEth/EVMYulLean", # irrelevant
    "rwbarton/advent-of-lean-4", # irrelevant
    "leanprover-community/tutorials4", # irrelevant
    "haruhisa-enomoto/mathlib4-all-tactics", # irrelevant
]

repos = []  # stores the names of all the repos
lean_git_repos = []  # stores the LeanGitRepo objects
attempted_repos = set()

personal_access_token = os.environ.get("GITHUB_ACCESS_TOKEN")

# TODO: change these
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
    # if os.path.exists(repo_name):
    #     print(f"Deleting existing repository directory: {repo_name}")
    #     shutil.rmtree(repo_name)
    # subprocess.run(["git", "clone", repo_url, repo_name])
    # TODO: no need for this if we check for latest compatible commit later
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

def get_compatible_commit(url):
    try:
        process = subprocess.Popen(["git", "ls-remote", url], stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        latest_commit = re.split(r'\t+', stdout.decode('utf-8'))[0]
        logger.info(f"Latest commit: {latest_commit}")

        new_url = url.replace('.git', '')
        logger.info(f"Creating LeanGitRepo for {new_url}")
        repo = LeanGitRepo(new_url, latest_commit)
        logger.info(f"Getting config for {url}")
        config = repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        if generate_benchmark_lean4.is_supported_version(v):
            logger.info(f"Latest commit compatible for url {url}")
            return latest_commit, v

        logger.info(f"Searching for compatible commit for {url}")
        try:
            subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], 
                        check=True, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
            logger.info("Already in a Git repository")
        except subprocess.CalledProcessError:
            logger.info("Not in a Git repository. Initializing one.")
            subprocess.run(["git", "init"], check=True)
        
        process = subprocess.Popen(
            ["git", "fetch", "--depth=1000000", url],  # Fetch commits
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Fetching commits for {url}")
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Git fetch command failed: {stderr.decode('utf-8')}")
        logger.info(f"Fetched commits for {url}")
        process = subprocess.Popen(
            ["git", "log", "--format=%H", "FETCH_HEAD"],  # Get list of commits
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Getting list of commits for {url}")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Git log command failed: {stderr.decode('utf-8')}")
        commits = stdout.decode('utf-8').strip().split('\n')
        logger.info(f"Found {len(commits)} commits for {url}")
        for commit in commits:
            new_url = url.replace('.git', '')
            repo = LeanGitRepo(new_url, commit)
            config = repo.get_config("lean-toolchain")
            v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
            if generate_benchmark_lean4.is_supported_version(v):
                logger.info(f"Found compatible commit {commit} for {url}")
                return commit, v

        raise Exception("No compatible commit found")

    except Exception as e:
        logger.info(f"Error in get_compatible_commit: {str(e)}")
        return None, None

def find_and_save_compatible_commits(repo_info_file, lean_git_repos):
    updated_repos = []
    for repo in lean_git_repos:
        url = repo.url
        if not url.endswith('.git'):
            url = url + '.git'
        
        sha = None
        v = None
        if "mathlib4" in url:
            sha = "2b29e73438e240a427bcecc7c0fe19306beb1310"
            v = "v4.8.0"
        elif "SciLean" in url:
            sha = "22d53b2f4e3db2a172e71da6eb9c916e62655744"
            v = "v4.7.0"
        elif "pfr" in url:
            sha = "fa398a5b853c7e94e3294c45e50c6aee013a2687"
            v = "v4.8.0-rc1"
        else:
            sha, v = get_compatible_commit(url)
        if not sha:
            logger.info(f"Failed to find a compatible commit for {url}")
            continue
        
        updated_repos.append({"url": url.replace('.git', ''), "commit": sha, "version": v})
    
    with open(repo_info_file, 'w') as f:
        json.dump(updated_repos, f)
    
    return updated_repos

def search_github_repositories(language="Lean", num_repos=10):
    global attempted_repos
    """Search for the given number of repositories on GitHub that have the given language."""
    headers = {'Authorization': personal_access_token}
    query_params = {
        'q': f'language:{language}',
        'sort': 'stars',
        'order': 'desc',
        'per_page': 100,
    }
    
    cloned_count = 0
    page = 1

    while cloned_count < num_repos:
        query_params['page'] = page
        response = requests.get('https://api.github.com/search/repositories', headers=headers, params=query_params)
        
        if response.status_code == 200:
            repositories = response.json()['items']
            for repo in repositories:
                if cloned_count >= num_repos:
                    break
                repo_full_name = repo['full_name']
                logger.info(f"Processing {repo_full_name}")
                if repo_full_name not in known_repositories and repo_full_name not in attempted_repos:
                    name = None
                    try:
                        clone_url = repo['clone_url']
                        repo_name, sha = clone_repo(clone_url)
                        name = repo_name
                        url = clone_url.replace('.git', '')
                        lean_git_repo = LeanGitRepo(url, sha)
                        lean_git_repos.append(lean_git_repo)
                        repos.append(repo_full_name)
                        cloned_count += 1 # TODO: only increase if compatible commit found
                        logger.info(f"Cloned {repo_full_name}")
                    except Exception as e:
                        # shutil.rmtree(name)
                        # Note: some 404s happen since some repos don't have a lean-toolchain
                        # but still use Lean
                        logger.info(f"Failed to clone {repo_full_name} because of {e}")
                else:
                    logger.info(f"Skipping {repo_full_name} since it is a known repository")
            page += 1
        else:
            logger.info("Failed to search GitHub", response.status_code)
            break

        # Check if we've reached the end of the search results
        if len(repositories) < 100:
            break
    
    logger.info(f"Total repositories processed: {cloned_count}")


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
            if key in preds_map:
                pred = preds_map[key]
            else:
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


def load_fisher_information(file_path):
    try:
        with open(file_path, 'rb') as f:
            fisher_info = pickle.load(f)
        logger.info("Fisher Information successfully loaded.")
        return fisher_info
    except FileNotFoundError:
        logger.error(f"No Fisher Information file found at {file_path}.")
        # TODO: return mathlib fisher
        return None

def find_latest_checkpoint():
    """Finds the most recent checkpoint."""
    checkpoint_dir = RAID_DIR + "/" + CHECKPOINT_DIR
    all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not all_checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    logger.info(f"Using the latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def find_latest_fisher():
    """Finds the most recent Fisher Information Matrix."""
    fisher_dir = RAID_DIR + "/" + FISHER_DIR
    all_fisher = [os.path.join(fisher_dir, f) for f in os.listdir(fisher_dir) if f.endswith(".pkl")]
    if not all_fisher:
        raise FileNotFoundError("No Fisher Information Matrices found.")
    latest_fisher = max(all_fisher, key=os.path.getmtime)
    logger.info(f"Using the latest Fisher Information Matrix: {latest_fisher}")
    return latest_fisher

# def train_test_fisher(model_checkpoint_path, new_data_path, lambda_value, current_epoch, use_fisher, epochs_per_repo=1):
#     logger.info("Inside train_test_fisher")
#     logger.info(f"Starting training at epoch {current_epoch}")
#     seed_everything(3407)

#     ### PROGRESSIVE TRAINING
    
#     if not torch.cuda.is_available():
#         logger.warning("Indexing the corpus using CPU can be very slow.")
#         device = torch.device("cpu")
#     else:
#         device = torch.device("cuda")

#     # TODO: reduce repetition in code like this
#     config = {
#         "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
#         "lr": 1e-3,
#         "warmup_steps": 1000,
#         "max_seq_len": 512,
#         "num_retrieved": 100,
#     }

#     model = PremiseRetriever.load(
#         model_checkpoint_path, device, freeze=False, config=config
#     )
#     model.train()
#     logger.info(f"Loaded premise retriever at {model_checkpoint_path} and reset epoch count")

#     # load previous Fisher Information Matrix for current EWC
#     if use_fisher:
#         latest_fisher = find_latest_fisher()
#         fisher_info = load_fisher_information(latest_fisher)
#         model.set_fisher_info(fisher_info)
#         logger.info("Fisher Information Matrix loaded.")

#     # TODO: use the yaml file instead of repeating here, same throughout
#     model.set_lambda(lambda_value)
#     corpus_path = new_data_path + "/corpus.jsonl"
#     data_path = new_data_path + "/random"
#     print(f"Data path: {data_path}")
#     data_module = RetrievalDataModule(
#         data_path=data_path,
#         corpus_path=corpus_path,
#         num_negatives=3,
#         num_in_file_negatives=1,
#         model_name="google/byt5-small",
#         batch_size=BATCH_SIZE,
#         eval_batch_size=64,
#         max_seq_len=1024,
#         num_workers=4
#     )
#     data_module.setup(stage='fit')

#     dir_name = new_data_path.split("/")[-1]
#     filename_suffix = f"_lambda_{lambda_value}"
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=RAID_DIR + "/" + CHECKPOINT_DIR,
#         filename=dir_name + filename_suffix + "_{epoch}-{Recall@10_val:.2f}",
#         verbose=True,
#         save_top_k=1,
#         save_last=False,
#         monitor="Recall@10_val",
#         mode="max"
#     )
    
#     early_stop_callback = EarlyStopping(
#         monitor="Recall@10_val",
#         patience=5,
#         mode="max",
#         verbose=True
#     )

#     lr_monitor = LearningRateMonitor(logging_interval='step')

#     print(f"Training dataset size after load: {len(data_module.ds_train)}")
#     print(f"Validation dataset size after load: {len(data_module.ds_val)}")
#     print(f"Testing dataset size after load: {len(data_module.ds_pred)}")

#     # TODO: have separate ones and save the epoch and data point, same for fisher
#     mid_epoch_checkpoint_dir = os.path.join(RAID_DIR, "mid_epoch_checkpoints")
#     os.makedirs(mid_epoch_checkpoint_dir, exist_ok=True)
#     timed_checkpoint_callback = TimedCheckpoint(checkpoint_dir=mid_epoch_checkpoint_dir)

#     VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60  # 1 week
#     os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
#     os.environ['NCCL_TIMEOUT'] = str(VERY_LONG_TIMEOUT * 1000)

#     custom_log_dir = os.path.join(RAID_DIR, "lightning_logs", f"{dir_name}_{use_fisher}_lambda_{lambda_value}")
#     os.makedirs(custom_log_dir, exist_ok=True)

#     ddp_strategy = DDPStrategy(timeout=timedelta(seconds=VERY_LONG_TIMEOUT))
#     trainer = pl.Trainer(
#         accelerator="gpu",
#         gradient_clip_val=1.0,
#         precision="bf16-mixed",
#         strategy=ddp_strategy,
#         devices=4, # TODO: change for GPU
#         accumulate_grad_batches=4,
#         callbacks=[lr_monitor, checkpoint_callback, early_stop_callback, timed_checkpoint_callback],
#         max_epochs=current_epoch + epochs_per_repo,
#         log_every_n_steps=1,
#         num_sanity_val_steps=0,
#         default_root_dir=custom_log_dir,
#     )

#     logger.info(f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}")

#     try:
#         # if "mathlib4" not in new_data_path:
#         #     trainer.strategy.barrier()
#         #     trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)
#         #     trainer.strategy.barrier()
#         trainer.strategy.barrier()
#         trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)
#         trainer.strategy.barrier()
#     except Exception as e:
#         print(f"An error occurred during training: {str(e)}")
#         print(traceback.format_exc())

#     logger.info(f"Finished progressive training at epoch {trainer.current_epoch}")

#     ### TESTING FOR AVERAGE RECALL

#     # TODO: don't load corpus and reindex for every repo we use for average recall
#     # Load the best model checkpoint
#     best_model_path = checkpoint_callback.best_model_path
#     if best_model_path:
#         best_model = PremiseRetriever.load(best_model_path, device, freeze=False, config=config)
#     else:
#         logger.warning("No best model found. Using the last trained model.")
#         best_model = model
#         # TODO: change this to the last model trained later
#         if use_fisher:
#             best_model_path = RAID_DIR + "/" + CHECKPOINT_DIR + "/" + "merged_with_new_mathlib4_2b29e73438e240a427bcecc7c0fe19306beb1310_lambda_0.1_epoch=0-Recall@10_val=58.95.ckpt"
#         else:
#             best_model_path = RAID_DIR + "/" + CHECKPOINT_DIR + "/" + "merged_with_new_mathlib4_2b29e73438e240a427bcecc7c0fe19306beb1310_lambda_0.1_epoch=0-Recall@10_val=60.65.ckpt"
#     best_model.eval()

#     logger.info("Testing...")
#     total_R1, total_R10, total_MRR = [], [], []
#     dataset_path = RAID_DIR + "/" + DATA_DIR
#     testing_paths = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]
#     with open(EVAL_RESULTS_FILE_PATH, "a") as f:
#         f.write("\n\n\n")
#         f.write(f"Results for {dir_name} with lambda = {lambda_value}")
#     for data_path in testing_paths:
#         # TODO: remove this for tests that do not use merged dataset
#         if "merged" not in data_path:
#             continue
#         # subprocess.run(["python","retrieval/main.py", "predict", "--config", "retrieval/confs/cli_lean4_random.yaml", "--ckpt_path", model_checkpoint_path, "--data-path", data_path], check=True)
#         run_cli(best_model_path, data_path)
#         num_gpus = 4 # TODO: change for GPU
#         preds_map = {}
#         for gpu_id in range(num_gpus):
#             with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
#                 preds = pickle.load(f)
#                 preds_map.update(preds)

#         logger.info("Loaded the predictions pickle files")
#         data_path = os.path.join(data_path, "random", "test.json")
#         data = json.load(open(data_path))
#         logger.info(f"Evaluating on {data_path}")
#         R1, R10, MRR = _eval(data, preds_map)
#         logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
#         total_R1.append(R1)
#         total_R10.append(R10)
#         total_MRR.append(MRR)
#         with open(EVAL_RESULTS_FILE_PATH, "a") as f:
#             f.write(f"Intermediate results for {data_path}")
#             f.write(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

#     avg_R1 = np.mean(total_R1)
#     avg_R10 = np.mean(total_R10)
#     avg_MRR = np.mean(total_MRR)

#     logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

#     # Save average accuracies to a file
#     if not os.path.exists(EVAL_RESULTS_FILE_PATH):
#         open(EVAL_RESULTS_FILE_PATH, 'w').close()

#     with open(EVAL_RESULTS_FILE_PATH, "a") as f:
#         f.write("\n\n\n")
#         f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

    
#     ### FISHER INFORMATION MATRIX FOR NEXT EWC

#     # TODO: distributed later
#     # Switch to one GPU for calculating the Fisher Information Matrix
#     if use_fisher:
#         try:
#             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#             best_model.to(device)
#             train_dataloader = data_module.train_dataloader()
#             fisher_info = best_model.compute_fisher_information(train_dataloader, RAID_DIR + "/" + FISHER_DIR)
#             dir_path = RAID_DIR + "/" + FISHER_DIR
#             fisher_name = dir_path + "/" + dir_name + "_fisher_info.pkl"
#             with open(fisher_name, "wb") as f:
#                 pickle.dump(fisher_info, f)
#             logger.info(f"Fisher info saved to {fisher_name}")
#         except Exception as e:
#             print(f"An error occurred during fisher: {str(e)}")
#             print(traceback.format_exc())

#     # TODO: add anything else from yaml conf if needed

#     return model

def theorem_identifier(theorem: Theorem) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
    return (theorem.full_name, str(theorem.file_path), tuple(theorem.start), tuple(theorem.end))

def process_theorem_batch(theorem_batch, positions_batch, repo, db, prover, dynamic_database_json_path):
    lean_dojo_theorems = [t[1] for t in theorem_batch]
    results = prover.search_unordered(LeanGitRepo(repo.url, repo.commit), lean_dojo_theorems, positions_batch)
    
    # Create a mapping from LeanDojoTheorem to our Theorem
    theorem_map = {ldj_thm: thm for thm, ldj_thm in theorem_batch}
    
    for result in results:
        if isinstance(result, SearchResult):
            if result.theorem in theorem_map:
                theorem = theorem_map[result.theorem]
                if result.status == Status.PROVED:
                    logger.info(f"Proof found for {theorem.full_name}")
                    traced_tactics = [
                        AnnotatedTactic(
                            tactic=tactic,
                            annotated_tactic=(tactic, []),
                            state_before="",
                            state_after=""
                        ) for tactic in result.proof
                    ]
                    theorem.traced_tactics = traced_tactics
                    repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
                    db.update_repository(repo)
                    logger.info(f"Updated theorem {theorem.full_name} in the database")
                else:
                    logger.info(f"No proof found for {theorem.full_name}")
            else:
                logger.warning(f"Theorem not found in theorem_map: {result.theorem}")
        else:
            logger.warning(f"Unexpected result type")
    
    db.to_json(dynamic_database_json_path)

def save_progress(all_encountered_theorems):
    logger.info("Saving encountered theorems...")
    with open(ENCOUNTERED_THEOREMS_FILE, 'wb') as f:
        pickle.dump(all_encountered_theorems, f)

def load_encountered_theorems(file_path):
    all_encountered_theorems = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                if file_content:  # Check if the file is not empty
                    all_encountered_theorems = pickle.loads(file_content)
                else:
                    logger.warning(f"The file {file_path} is empty. Starting with an empty set.")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Error reading {file_path}: {e}. Starting with an empty set.")
        except Exception as e:
            logger.error(f"Unexpected error when reading {file_path}: {e}. Starting with an empty set.")
    else:
        logger.info(f"The file {file_path} does not exist. Starting with an empty set.")
    
    return all_encountered_theorems

def prove_sorry_theorems(db: DynamicDatabase, prover: DistributedProver, dynamic_database_json_path, repos_to_include: Optional[List[Tuple[str, str]]] = None, batch_size: int = 12):
    repos_to_process = db.repositories if repos_to_include is None else [
        repo for repo in db.repositories if (repo.url, repo.commit) in repos_to_include
    ]

    # To avoid proving the same theorem multiple times, potentially from different versions of the
    # same repo, we sort the repositories
    repos_to_process.sort(key=lambda r: r.metadata['date_processed'], reverse=True)

    processed_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
    all_encountered_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
    last_save_time = datetime.datetime.now()
    save_interval = timedelta(minutes=30)

    # Load previously encountered theorems
    all_encountered_theorems = load_encountered_theorems(ENCOUNTERED_THEOREMS_FILE)

    for repo in repos_to_process:
        sorry_theorems = repo.sorry_theorems_unproved
        repo_url = repo.url
        repo_commit = repo.commit

        logger.info(f"Found {len(sorry_theorems)} sorry theorems to prove")

        theorem_batch = []
        positions_batch = []
    
        for theorem in tqdm(sorry_theorems, desc=f"Processing theorems from {repo.name}", unit="theorem"):
            # Ignore sorry theorems from the repo's dependencies
            if theorem.url != repo_url or theorem.commit != repo_commit:
                continue

            theorem_id = theorem_identifier(theorem)

            if theorem_id in all_encountered_theorems:
                logger.info(f"Skipping already encountered theorem: {theorem.full_name}")
                continue

            all_encountered_theorems.add(theorem_id)
            if theorem_id in processed_theorems:
                logger.info(f"Skipping already processed theorem: {theorem.full_name}")
                continue

            processed_theorems.add(theorem_id)
            
            logger.info(f"Searching for proof for {theorem.full_name}")
            logger.info(f"Position: {theorem.start}")

            # Convert our Theorem to LeanDojo Theorem
            lean_dojo_theorem = LeanDojoTheorem(
                repo=LeanGitRepo(repo_url, repo_commit),
                file_path=theorem.file_path,
                full_name=theorem.full_name
            )

            theorem_batch.append((theorem, lean_dojo_theorem))
            positions_batch.append(Pos(*theorem.start))

            if len(theorem_batch) == batch_size:
                process_theorem_batch(theorem_batch, positions_batch, repo, db, prover, dynamic_database_json_path)
                theorem_batch = []
                positions_batch = []

            current_time = datetime.datetime.now()
            if current_time - last_save_time >= save_interval:
                save_progress(all_encountered_theorems)
                last_save_time = current_time
        
        # Process any remaining theorems in the last batch
        if theorem_batch:
            process_theorem_batch(theorem_batch, positions_batch, repo, db, prover, dynamic_database_json_path)

    save_progress(all_encountered_theorems)
    logger.info("Finished attempting to prove sorry theorems")

# class TimedCheckpoint(Callback):
#     def __init__(self, checkpoint_dir, interval=timedelta(hours=4)):
#         self.checkpoint_dir = checkpoint_dir
#         self.interval = interval
#         self.last_checkpoint_time = datetime.datetime.now()

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         now = datetime.datetime.now()
#         if now - self.last_checkpoint_time >= self.interval:
#             epoch = trainer.current_epoch
#             global_step = trainer.global_step
#             checkpoint_path = os.path.join(self.checkpoint_dir, f'mid_epoch_checkpoint_epoch_{epoch}_step_{global_step}.ckpt')
#             trainer.save_checkpoint(checkpoint_path)
#             self.last_checkpoint_time = now
#             logger.info(f"Mid-epoch checkpoint saved at {checkpoint_path}")

def add_repo_to_database(dynamic_database_json_path, repo, db):
    # Prepare the data necessary to add this repo to the dynamic database
    url = repo.url
    if not url.endswith('.git'):
        url = url + '.git'
    logger.info(f"Processing {url}")
    # TODO: remove later
    # TODO: duplicate work of the main function
    # sha, v = get_compatible_commit(url)
    sha = None
    v = None
    if "mathlib4" in url:
        sha = "2b29e73438e240a427bcecc7c0fe19306beb1310"
        v = "v4.8.0"
    elif "SciLean" in url:
        sha = "22d53b2f4e3db2a172e71da6eb9c916e62655744"
        v = "v4.7.0"
    elif "pfr" in url:
        sha = "fa398a5b853c7e94e3294c45e50c6aee013a2687"
        v = "v4.8.0-rc1"
    else:
        sha, v = get_compatible_commit(url)
    if not sha:
        logger.info(f"Failed to find a compatible commit for {url}")
        return None
    logger.info(f"Found compatible commit {sha} for {url}")
    logger.info(f"Lean version: {v}")
    url = url.replace('.git', '')
    repo = LeanGitRepo(url, sha)
    dir_name = repo.url.split("/")[-1] + "_" + sha
    dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name
    logger.info(f"Generating benchmark at {dst_dir}")
    # traced_repo, _, _, total_theorems = generate_benchmark_lean4.main(repo.url, sha, dst_dir)
    # if not traced_repo:
    #     logger.info(f"Failed to trace {url}")
    #     return None
    # if total_theorems < 3 * BATCH_SIZE:  # Should be enough theorems for train/val/test
    #     logger.info(f"No theorems found in {url}")
    #     return None
    logger.info(f"Finished generating benchmark at {dst_dir}")

    # Add the new repo to the dynamic database
    config = repo.get_config("lean-toolchain")
    v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
    theorems_folder = dst_dir + "/random"
    premise_files_corpus = dst_dir + "/corpus.jsonl"
    files_traced = dst_dir + "/traced_files.jsonl"
    pr_url = None
    data = {
        "url": repo.url,
        "name": "/".join(repo.url.split("/")[-2:]),
        "commit": repo.commit,
        "lean_version": v,
        "lean_dojo_version": lean_dojo.__version__,
        "metadata": {
            "date_processed": datetime.datetime.now(),
        },
        "theorems_folder": theorems_folder,
        "premise_files_corpus": premise_files_corpus,
        "files_traced": files_traced,
        "pr_url": pr_url
    }
    
    repo = Repository.from_dict(data)
    logger.info("Before adding new repo:")
    db.print_database_contents()
    db.add_repository(repo)
    logger.info("After adding new repo:")
    db.print_database_contents()
    db.to_json(dynamic_database_json_path)
    return "Done"

# def retrieve_proof(run_progressive_training, use_fisher, single_repo, curriculum_learning, dynamic_database_json_path, repo, sha, lambda_value, current_epoch, epochs_per_repo, db):
#     global repos_for_merged_dataset
#     # TODO: update comments throughout
#     """
#     This method does the following:
#     1. Check if the given repo is supported.
#     2. Trace the repo.
#     3. Generate a corpus of the repo's premises.
#     4. Search for proofs for theorems with `sorry` in them.
#     """
#     if single_repo:
#         repos_for_merged_dataset = []

#     if not curriculum_learning:
#         result = add_repo_to_database(dynamic_database_json_path, repo, db)
#         if result is None:
#             return None

#     # Generate a new dataset from the dynamic database.
#     # The user can choose to generate a dataset from the entire dynamic database or a subset of it.
#     dir_name = repo.url.split("/")[-1] + "_" + sha
#     # TODO: don't always do merged_, if we change this then change the if condition in average test accordingly
#     dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_with_new_{dir_name}"
#     if (repo.url, repo.commit) not in repos_for_merged_dataset:
#         logger.info("Adding repo to repos_for_merged_dataset")
#         repos_for_merged_dataset.append((repo.url, repo.commit))
#     else:
#         logger.info("Repo already in repos_for_merged_dataset")
#     db.generate_merged_dataset(dst_dir, repos_for_merged_dataset)
#     # TODO: reduce repition later with all path
#     dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + f"merged_with_new_{dir_name}"

#     model_checkpoint_path = None
#     if run_progressive_training:
#         try:
#             model_checkpoint_path = find_latest_checkpoint()
#             logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
#         except FileNotFoundError as e:
#             logger.error(str(e))
#             return None
        
#         # Train the model on the new dataset that we generated from the dynamic database.
#         train_test_fisher(model_checkpoint_path, dst_dir, lambda_value, current_epoch, use_fisher, epochs_per_repo)
#     else:
#         model_checkpoint_path = f"{RAID_DIR}/checkpoints_PT_full_merge_each_time_ewc/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"

#     # Set up the prover
#     use_vllm = False
#     corpus_path = dst_dir + "/corpus.jsonl"
#     tactic = None  # `None` since we are not using a fixed tactic generator
#     module = None  # `None` since we are not using a fixed tactic generator
#     num_workers = 4 # TODO: do everywhere if good
#     num_gpus = 4 # TODO: change for GPU
#     timeout = 600
#     max_expansions = None
#     num_sampled_tactics = 64
#     debug = False
#     ckpt_path = f"{RAID_DIR}/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_lightning.ckpt"
#     prover = DistributedProver(
#         use_vllm,
#         ckpt_path,
#         corpus_path,
#         tactic,
#         module,
#         num_workers,
#         num_gpus=num_gpus,
#         timeout=timeout,
#         max_expansions=max_expansions,
#         num_sampled_tactics=num_sampled_tactics,
#         raid_dir=RAID_DIR,
#         checkpoint_dir=CHECKPOINT_DIR,
#         debug=debug,
#         run_progressive_training=run_progressive_training
#     )

#     # Prove sorry theorems
#     prove_sorry_theorems(db, prover, dynamic_database_json_path, repos_for_merged_dataset)
#     db.to_json(dynamic_database_json_path)

#     logger.info("Finished searching for proofs of sorry theorems")

#     # TODO: need to return proofs
#     proofs = []
#     return proofs

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

    logger.info("Finished replacing sorries with proofs!")

def calculate_difficulty(theorem: Theorem) -> Union[float, None]:
    proof_steps = theorem.traced_tactics
    if any('sorry' in step.tactic for step in proof_steps):
        return float('inf')  # Hard (no proof)
    if len(proof_steps) == 0:
        return None  # To be distributed later
    return math.exp(len(proof_steps))

def categorize_difficulty(difficulty: Union[float, None], percentiles: List[float]) -> str:
    if difficulty is None:
        return "To_Distribute"
    if difficulty == float('inf'):
        return "Hard (No proof)"
    elif difficulty <= percentiles[0]:
        return "Easy"
    elif difficulty <= percentiles[1]:
        return "Medium"
    else:
        return "Hard"

def sort_repositories_by_difficulty(db: DynamicDatabase) -> List[Repository]:
    difficulties_by_repo = defaultdict(list)
    all_difficulties = []

    print("Ready to calculate difficulties of all theorems")
    for repo in db.repositories:
        print(f"Starting {repo.name}")
        for theorem in repo.get_all_theorems:
            difficulty = calculate_difficulty(theorem)
            theorem.difficulty_rating = difficulty
            difficulties_by_repo[repo].append((theorem.full_name, str(theorem.file_path), tuple(theorem.start), tuple(theorem.end), difficulty))
            if difficulty is not None:
                all_difficulties.append(difficulty)
            
        db.update_repository(repo)
        print(f"Finished {repo.name}")

    percentiles = np.percentile(all_difficulties, [33, 67])

    categorized_theorems = defaultdict(lambda: defaultdict(list))

    print("Ready to categorize theorems")
    for repo, theorems in difficulties_by_repo.items():
        print(f"Starting {repo.name}")
        for theorem_name, file_path, start, end, difficulty in theorems:
            category = categorize_difficulty(difficulty, percentiles)
            categorized_theorems[repo][category].append((theorem_name, file_path, start, end, difficulty))
        print(f"Finished {repo.name}")

    print("Distributed theorems with no proofs")
    for repo in categorized_theorems:
        print(f"Starting {repo.name}")
        to_distribute = categorized_theorems[repo]["To_Distribute"]
        chunk_size = len(to_distribute) // 3
        for i, category in enumerate(["Easy", "Medium", "Hard"]):
            start = i * chunk_size
            end = start + chunk_size if i < 2 else None
            categorized_theorems[repo][category].extend(to_distribute[start:end])
        del categorized_theorems[repo]["To_Distribute"]
        print(f"Finished {repo.name}")

    # # Sort repositories based on the number of easy theorems
    sorted_repos = sorted(categorized_theorems.keys(), key=lambda r: len(categorized_theorems[r]["Easy"]), reverse=True)

    # Calculate percentage of easy theorems for each repository
    # repo_easy_percentages = {}
    # for repo in categorized_theorems:
    #     total_theorems = sum(len(theorems) for theorems in categorized_theorems[repo].values())
    #     easy_theorems = len(categorized_theorems[repo]["Easy"])
    #     easy_percentage = (easy_theorems / total_theorems) * 100 if total_theorems > 0 else 0
    #     repo_easy_percentages[repo] = easy_percentage

    # # Sort repositories based on the percentage of easy theorems
    # sorted_repos = sorted(categorized_theorems.keys(), key=lambda r: repo_easy_percentages[r], reverse=True)

    # return sorted_repos, categorized_theorems, percentiles, repo_easy_percentages
    return sorted_repos, categorized_theorems, percentiles

def save_sorted_repos(sorted_repos: List[Repository], file_path: str):
    sorted_repo_data = [
        {
            "url": repo.url,
            "commit": repo.commit,
            "name": repo.name
        } for repo in sorted_repos
    ]
    with open(file_path, 'w') as f:
        json.dump(sorted_repo_data, f, indent=2)

def load_sorted_repos(file_path: str) -> List[Tuple[str, str, str]]:
    with open(file_path, 'r') as f:
        sorted_repo_data = json.load(f)
    return [(repo["url"], repo["commit"], repo["name"]) for repo in sorted_repo_data]

def write_skip_file(repo_url):
    skip_file_path = os.path.join(RAID_DIR, DATA_DIR, "skip_repo.txt")
    with open(skip_file_path, 'w') as f:
        f.write(repo_url)

def should_skip_repo():
    skip_file_path = os.path.join(RAID_DIR, DATA_DIR, "skip_repo.txt")
    if os.path.exists(skip_file_path):
        with open(skip_file_path, 'r') as f:
            repo_url = f.read().strip()
        return True, repo_url
    return False, None

# TODO: incorporate latest changes from ReProver repo
def main():
    """The main function that drives the bot."""
    global repos_for_merged_dataset
    global lean_git_repos
    try:
        # Configure these parameters!
        current_epoch = 0
        epochs_per_repo = 1
        run_progressive_training = True
        # run_progressive_training = False
        # use_fisher = True
        use_fisher = False
        single_repo = True
        # single_repo = False
        start_with_pfr = False
        # start_with_pfr = True
        # curriculum_learning = False
        curriculum_learning = True
        num_repos = 15
        dynamic_database_json_path = RAID_DIR + "/" + DB_FILE_NAME
        
        lambdas = None
        if run_progressive_training:
            logger.info("Running progressive training")
            lambdas = [0.1]
        else:
            logger.info("Running retrieval baseline")
            lambdas = [0.0]

        logger.info("Configuring LeanDojo...")
        generate_benchmark_lean4.configure_leandojo()
        logger.info("LeanDojo configured")

        is_main_process = int(os.environ.get('LOCAL_RANK', '0')) == 0

        if is_main_process:
            logger.info("Starting the main process")
            if not os.path.exists(dynamic_database_json_path) or os.path.getsize(dynamic_database_json_path) == 0:
                # File doesn't exist or is empty, initialize it
                logger.info(f"Initializing new database at {dynamic_database_json_path}")
                db = DynamicDatabase()
                db.to_json(dynamic_database_json_path)
            else:
                try:
                    logger.info(f"Loading database from {dynamic_database_json_path}")
                    db = DynamicDatabase.from_json(dynamic_database_json_path)
                    logger.info(f"Loaded database from {dynamic_database_json_path}")
                except json.JSONDecodeError:
                    # If there's an error decoding the JSON, initialize a new database
                    logger.warning(f"Error decoding JSON from {dynamic_database_json_path}. Initializing new database.")
                    db = DynamicDatabase()
                    db.to_json(dynamic_database_json_path)

        logger.info(f"Found {num_repos} repositories")

        if curriculum_learning:
            logger.info("Starting curriculum learning")
            repo_info_file = f"{RAID_DIR}/{DATA_DIR}/repo_info_compatible.json"  # TODO: make constnat?
            # if is_main_process:

                # clone_url = "https://github.com/yangky11/miniF2F-lean4"
                # commit = "9e445f5435407f014b88b44a98436d50dd7abd00"
                # url = clone_url.replace('.git', '')
                # lean_git_repo = LeanGitRepo(url, commit)
                # lean_git_repos.append(lean_git_repo)

                # search_github_repositories("Lean", num_repos)
                # for i in range(len(lean_git_repos)):
                #     repo = lean_git_repos[i]
                #     logger.info(f"Processing {repo.url}")
                #     result = add_repo_to_database(dynamic_database_json_path, repo, db)
                #     if result is not None:
                #         logger.info(f"Successfully added repo {repo.url}")                    
                # logger.info(f"Successfully added {num_repos} repositories to the database")
                
                # sorted_repos, categorized_theorems, percentiles = sort_repositories_by_difficulty(db)
                # print("Sorted repositories. Saving now...")
                # db.to_json(dynamic_database_json_path)
                # save_sorted_repos(sorted_repos, "sorted_repos.json")
                # print("Summary of theorem difficulties by URL:")
                # for repo in sorted_repos:
                #     print(f"\nURL: {repo.url}")
                #     for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                #         theorems = categorized_theorems[repo][category]
                #         print(f"  {category}: {len(theorems)} theorems")
                #         if theorems:
                #             sorted_theorems = sorted(theorems, key=lambda x: x[2] if x[2] is not None else -float('inf'), reverse=True)[:3]
                #             for name, path, start, end, diff in sorted_theorems:
                #                 diff_str = f"{diff:.2f}" if diff is not None else "N/A"
                #                 print(f"    - {name} (File: {path}, Difficulty: {diff_str})")

                # sorted_repos, categorized_theorems, percentiles, repo_easy_percentages = sort_repositories_by_difficulty(db)
                # print("Sorted repositories. Saving now...")
                # db.to_json(dynamic_database_json_path)
                # save_sorted_repos(sorted_repos, "sorted_repos.json")
                # print("Summary of theorem difficulties by URL:")
                # for repo in sorted_repos:
                #     print(f"\nURL: {repo.url}")
                #     total_theorems = sum(len(theorems) for theorems in categorized_theorems[repo].values())
                #     for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                #         theorems = categorized_theorems[repo][category]
                #         percentage = (len(theorems) / total_theorems) * 100 if total_theorems > 0 else 0
                #         print(f"  {category}: {len(theorems)} theorems ({percentage:.2f}%)")
                #         if theorems:
                #             sorted_theorems = sorted(theorems, key=lambda x: x[4] if x[4] is not None else -float('inf'), reverse=True)[:3]
                #             for name, path, start, end, diff in sorted_theorems:
                #                 diff_str = f"{diff:.2f}" if diff is not None else "N/A"
                #                 print(f"    - {name} (File: {path}, Difficulty: {diff_str})")
                #     print(f"  Easy Theorem Percentage: {repo_easy_percentages[repo]:.2f}%")

                # print("\nOverall Statistics:")
                # total_theorems = sum(len(theorems) for categories in categorized_theorems.values() for theorems in categories.values())
                # for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                #     count = sum(len(categories[category]) for categories in categorized_theorems.values())
                #     percentage = (count / total_theorems) * 100
                #     print(f"{category}: {count} theorems ({percentage:.2f}%)")

                # print(f"\nPercentile thresholds: Easy <= {percentiles[0]:.2f}, Medium <= {percentiles[1]:.2f}, Hard > {percentiles[1]:.2f}")
            
                # logger.info("Finding compatible repositories...")
                # updated_repos = find_and_save_compatible_commits(repo_info_file, sorted_repos)
                # lean_git_repos = [LeanGitRepo(repo['url'], repo['commit']) for repo in updated_repos]
                # logger.info("Finished finding compatible repositories")
                # return

            # All processes wait for the file to be created and then read from it
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    with open(repo_info_file, 'r') as f:
                        repo_info = json.load(f)
                    break
                except (json.JSONDecodeError, FileNotFoundError):
                    if attempt == max_attempts - 1:
                        raise Exception("Failed to read repository information after multiple attempts")
                    time.sleep(1)
                
            lean_git_repos = [LeanGitRepo(info['url'].replace('.git', ''), info['commit']) for info in repo_info]

            for i in range(num_repos):
                for lambda_value in lambdas:
                    logger.info(f"length of lean_git_repos: {len(lean_git_repos)}")
                    logger.info(f"i: {i}")
                    repo = lean_git_repos[i]
                    sha = repo.commit
                    dir_name = repo.url.split("/")[-1] + "_" + sha
                    result = True
                    if is_main_process:
                        logger.info("Main process")
                        logger.info(f"Using lambda = {lambda_value}")
                        logger.info(f"Processing {repo.url}")

                        if single_repo:
                            repos_for_merged_dataset = []

                        dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_with_new_{dir_name}"
                        if (repo.url, repo.commit) not in repos_for_merged_dataset:
                            logger.info("Adding repo to repos_for_merged_dataset")
                            repos_for_merged_dataset.append((repo.url, repo.commit))
                        else:
                            logger.info("Repo already in repos_for_merged_dataset")

                        if "miniF2F" not in repo.url:
                            db.generate_merged_dataset(dst_dir, repos_for_merged_dataset)
                    
                    # TODO: reduce repition later with all path
                    dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + f"merged_with_new_{dir_name}"
                    new_data_path = dst_dir

                    # All GPUs
                    logger.info("All GPUs")
                    model_checkpoint_path = None
                    best_model = None
                    data_module = None
                    if run_progressive_training:
                        try:
                            model_checkpoint_path = find_latest_checkpoint()
                            logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
                        except FileNotFoundError as e:
                            logger.error(str(e))
                            model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                        
                        # Train the model on the new dataset that we generated from the dynamic database.
                        logger.info("Inside train_test_fisher")
                        logger.info(f"Starting training at epoch {current_epoch}")
                        seed_everything(3407)

                        ### PROGRESSIVE TRAINING
                        
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

                        model = PremiseRetriever.load(
                            model_checkpoint_path, device, freeze=False, config=config
                        )
                        model.train()
                        logger.info(f"Loaded premise retriever at {model_checkpoint_path}")

                        # load previous Fisher Information Matrix for current EWC
                        if use_fisher:
                            latest_fisher = find_latest_fisher()
                            fisher_info = load_fisher_information(latest_fisher)
                            model.set_fisher_info(fisher_info)
                            logger.info("Fisher Information Matrix loaded.")

                        # TODO: use the yaml file instead of repeating here, same throughout
                        dir_name = new_data_path.split("/")[-1]
                        filename_suffix = f"_lambda_{lambda_value}"
                        checkpoint_callback = ModelCheckpoint(
                            dirpath=RAID_DIR + "/" + CHECKPOINT_DIR,
                            filename=dir_name + filename_suffix + "_{epoch}-{Recall@10_val:.2f}",
                            verbose=True,
                            save_top_k=-1,  # Save all checkpoints
                            every_n_epochs=1,  # Save every epoch (which is just once in this case)
                            monitor="Recall@10_val",
                            mode="max"
                        )
                        
                        early_stop_callback = EarlyStopping(
                            monitor="Recall@10_val",
                            patience=5,
                            mode="max",
                            verbose=True
                        )

                        lr_monitor = LearningRateMonitor(logging_interval='step')

                        # mid_epoch_checkpoint_dir = os.path.join(RAID_DIR, f"mid_epoch_checkpoints_{current_epoch}_{dir_name}_{use_fisher}_lambda_{lambda_value}")
                        # os.makedirs(mid_epoch_checkpoint_dir, exist_ok=True)
                        # timed_checkpoint_callback = TimedCheckpoint(checkpoint_dir=mid_epoch_checkpoint_dir)

                        VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60 * 52  # 1 year
                        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
                        os.environ['NCCL_TIMEOUT'] = str(VERY_LONG_TIMEOUT * 1000)

                        custom_log_dir = os.path.join(RAID_DIR, "lightning_logs", f"{dir_name}_{use_fisher}_lambda_{lambda_value}")
                        os.makedirs(custom_log_dir, exist_ok=True)

                        ddp_strategy = DDPStrategy(timeout=timedelta(seconds=VERY_LONG_TIMEOUT))
                        trainer = pl.Trainer(
                            accelerator="gpu",
                            gradient_clip_val=1.0,
                            precision="bf16-mixed",
                            strategy=ddp_strategy,
                            devices=4, # TODO: change for GPU
                            accumulate_grad_batches=4,
                            # callbacks=[lr_monitor, checkpoint_callback, early_stop_callback, timed_checkpoint_callback],
                            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
                            max_epochs=current_epoch + epochs_per_repo,
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            default_root_dir=custom_log_dir,
                        )

                        logger.info("right before barrier for data module")
                        trainer.strategy.barrier()
                        should_skip, skip_repo_url = should_skip_repo()
                        if should_skip:
                            logger.info(f"Skipping repository {skip_repo_url} due to preprocessing issues")
                            trainer.strategy.barrier()
                            if is_main_process:
                                logger.info("Removing skip file")
                                skip_file_path = os.path.join(RAID_DIR, DATA_DIR, "skip_repo.txt")
                                os.remove(skip_file_path)
                            continue

                        model.set_lambda(lambda_value)
                        corpus_path = new_data_path + "/corpus.jsonl"
                        data_path = new_data_path + "/random"
                        logger.info(f"Data path: {data_path}")
                        data_module = RetrievalDataModule(
                            data_path=data_path,
                            corpus_path=corpus_path,
                            num_negatives=3,
                            num_in_file_negatives=1,
                            model_name="google/byt5-small",
                            batch_size=BATCH_SIZE,
                            eval_batch_size=64,
                            max_seq_len=1024,
                            num_workers=4
                        )
                        data_module.setup(stage='fit')

                        logger.info(f"Training dataset size after load: {len(data_module.ds_train)}")
                        logger.info(f"Validation dataset size after load: {len(data_module.ds_val)}")
                        logger.info(f"Testing dataset size after load: {len(data_module.ds_pred)}")

                        logger.info(f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}")

                        try:
                            logger.info("hit the barrier before training")
                            trainer.strategy.barrier()
                            trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)
                            logger.info("hit the barrier after training")
                            trainer.strategy.barrier()
                        except Exception as e:
                            print(f"An error occurred during training: {str(e)}")
                            print(traceback.format_exc())

                        logger.info(f"Finished progressive training at epoch {trainer.current_epoch}")

                        ### TESTING FOR AVERAGE RECALL

                        # TODO: don't load corpus and reindex for every repo we use for average recall
                        try:
                            best_model_path = find_latest_checkpoint()
                            logger.info(f"Found latest checkpoint: {best_model_path}")
                            best_model = PremiseRetriever.load(best_model_path, device, freeze=False, config=config)
                        except FileNotFoundError as e:
                            logger.error(f"No checkpoint found: {str(e)}")
                            logger.warning("Using the current model state.")
                            best_model = model

                        best_model.eval()

                        # if "Foundation" not in repo.url:
                        #     logger.info("Testing...")
                        #     total_R1, total_R10, total_MRR = [], [], []
                        #     dataset_path = RAID_DIR + "/" + DATA_DIR
                        #     testing_paths = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]
                        #     if is_main_process:
                        #         with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                        #             f.write("\n\n\n")
                        #             f.write(f"Results for {dir_name} with lambda = {lambda_value}")
                        #     for data_path in testing_paths:
                        #         # TODO: remove this for tests that do not use merged dataset
                        #         if "merged" not in data_path:
                        #             continue
                        #         # subprocess.run(["python","retrieval/main.py", "predict", "--config", "retrieval/confs/cli_lean4_random.yaml", "--ckpt_path", model_checkpoint_path, "--data-path", data_path], check=True)
                        #         run_cli(best_model_path, data_path)
                        #         if is_main_process:
                        #             num_gpus = 4 # TODO: change for GPU
                        #             preds_map = {}
                        #             for gpu_id in range(num_gpus):
                        #                 with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
                        #                     preds = pickle.load(f)
                        #                     preds_map.update(preds)

                        #             logger.info("Loaded the predictions pickle files")
                        #             data_path = os.path.join(data_path, "random", "test.json")
                        #             data = json.load(open(data_path))
                        #             logger.info(f"Evaluating on {data_path}")
                        #             R1, R10, MRR = _eval(data, preds_map)
                        #             logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
                        #             total_R1.append(R1)
                        #             total_R10.append(R10)
                        #             total_MRR.append(MRR)
                        #             with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                        #                 f.write("\n\n\n")
                        #                 f.write(f"Intermediate results for {data_path}")
                        #                 f.write("\n\n\n")
                        #                 f.write(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

                        #     if is_main_process:
                        #         avg_R1 = np.mean(total_R1)
                        #         avg_R10 = np.mean(total_R10)
                        #         avg_MRR = np.mean(total_MRR)

                        #         logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

                        #         if not os.path.exists(EVAL_RESULTS_FILE_PATH):
                        #             open(EVAL_RESULTS_FILE_PATH, 'w').close()

                        #         with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                        #             f.write("\n\n\n")
                        #             f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")
                    else:
                        model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                        if result is None:
                            logger.info(f"Skipping repository {repo.url} due to preprocessing issues")
                            continue

                    if is_main_process and run_progressive_training and use_fisher:
                        logger.info("Calculating Fisher Information Matrix for EWC")
                        ### FISHER INFORMATION MATRIX FOR NEXT EWC

                        # Switch to one GPU for calculating the Fisher Information Matrix
                        # TODO: barrier here
                        try:
                            # TODO: have separate intermediate checkpoints and save the epoch and data point, same for fisher
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            best_model.to(device)
                            train_dataloader = data_module.train_dataloader()
                            fisher_info = best_model.compute_fisher_information(train_dataloader, RAID_DIR + "/" + FISHER_DIR)
                            dir_path = RAID_DIR + "/" + FISHER_DIR
                            fisher_name = dir_path + "/" + dir_name + "_fisher_info.pkl"
                            with open(fisher_name, "wb") as f:
                                pickle.dump(fisher_info, f)
                            logger.info(f"Fisher info saved to {fisher_name}")
                        except Exception as e:
                            print(f"An error occurred during fisher: {str(e)}")
                            print(traceback.format_exc())

                    if is_main_process:
                        logger.info("Starting the prover")

                        if ray.is_initialized():
                            logger.info("Shutting down Ray before proving")
                            ray.shutdown()

                        # Set up the prover
                        use_vllm = False
                        corpus_path = dst_dir + "/corpus.jsonl"
                        tactic = None  # `None` since we are not using a fixed tactic generator
                        module = None  # `None` since we are not using a fixed tactic generator
                        num_workers = 4
                        num_gpus = 4 # TODO: change for GPU
                        timeout = 600
                        max_expansions = None
                        num_sampled_tactics = 64
                        debug = False
                        ckpt_path = f"{RAID_DIR}/model_lightning.ckpt"
                        prover = DistributedProver(
                            use_vllm,
                            ckpt_path,
                            corpus_path,
                            tactic,
                            module,
                            num_workers,
                            num_gpus=num_gpus,
                            timeout=timeout,
                            max_expansions=max_expansions,
                            num_sampled_tactics=num_sampled_tactics,
                            raid_dir=RAID_DIR,
                            checkpoint_dir=CHECKPOINT_DIR,
                            debug=debug,
                            run_progressive_training=run_progressive_training
                        )

                        # Prove sorry theorems
                        prove_sorry_theorems(db, prover, dynamic_database_json_path, repos_for_merged_dataset)
                        db.to_json(dynamic_database_json_path)

                        logger.info("Finished searching for proofs of sorry theorems")

                        if ray.is_initialized():
                            logger.info("Shutting down Ray after proving")
                            ray.shutdown()

                        # TODO: need to return proofs
                        # proofs = []
                        # Uncomment if you would like to contribute back to the repos!
                        # else:
                        #     base_branch = get_default_branch(repo_no_dir)
                        #     subprocess.run(["git", "-C", repo, "fetch", "origin", base_branch], check=True)
                        #     subprocess.run(["git", "-C", repo, "checkout", base_branch], check=True)
                        #     subprocess.run(["git", "-C", repo, "pull", "origin", base_branch], check=True)
                        #     create_or_switch_branch(repo, TMP_BRANCH, base_branch)
                        #     replace_sorry_with_proof(proofs)
                        #     committed = commit_changes(repo, COMMIT_MESSAGE)
                        #     if committed:
                        #         push_changes(repo, TMP_BRANCH)
                        #         url = str(create_pull_request(repo_no_dir, PR_TITLE, PR_BODY, TMP_BRANCH))
                        #         # TODO: add the PR URL to the database
                        #     shutil.rmtree(repo)
                    
                    logger.info("Finished processing the repository")
                    current_epoch += epochs_per_repo
                    logger.info(f"current epoch: {current_epoch}")

        else:
            logger.info("Starting without curriculum learning")
            repo_info_file = f"{RAID_DIR}/{DATA_DIR}/repo_info_compatible.json"  # TODO: make constnat?
            # if is_main_process:
                # search_github_repositories("Lean", num_repos)

                # clone_url = "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd.git"
                # commit = "29baddd685660b5fedd7bd67f9916ae24253d566"
                # url = clone_url.replace('.git', '')
                # lean_git_repo = LeanGitRepo(url, commit)
                # lean_git_repos.append(lean_git_repo)

                # clone_url = "https://github.com/teorth/pfr.git"
                # commit = "fa398a5b853c7e94e3294c45e50c6aee013a2687"
                # url = clone_url.replace('.git', '')
                # lean_git_repo = LeanGitRepo(url, commit)
                # lean_git_repos.append(lean_git_repo)

                # clone_url = "https://github.com/avigad/mathematics_in_lean_source.git"
                # commit = "cfe61bc71b5ea501f89df36c945949a1febf5e75"
                # url = clone_url.replace('.git', '')
                # lean_git_repo = LeanGitRepo(url, commit)
                # lean_git_repos.append(lean_git_repo)

                # logger.info("Finding compatible repositories...")
                # updated_repos = find_and_save_compatible_commits(repo_info_file, lean_git_repos)
                # lean_git_repos = [LeanGitRepo(repo['url'], repo['commit']) for repo in updated_repos]
                # logger.info("Finished finding compatible repositories")

            # All processes wait for the file to be created and then read from it
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    with open(repo_info_file, 'r') as f:
                        repo_info = json.load(f)
                    break
                except (json.JSONDecodeError, FileNotFoundError):
                    if attempt == max_attempts - 1:
                        raise Exception("Failed to read repository information after multiple attempts")
                    time.sleep(1)
                
            lean_git_repos = [LeanGitRepo(info['url'].replace('.git', ''), info['commit']) for info in repo_info]

            for i in range(num_repos):
                for lambda_value in lambdas:
                    logger.info(f"length of lean_git_repos: {len(lean_git_repos)}")
                    logger.info(f"i: {i}")
                    repo = lean_git_repos[i]
                    sha = repo.commit
                    dir_name = repo.url.split("/")[-1] + "_" + sha
                    result = True
                    if is_main_process:
                        logger.info("Main process")
                        logger.info(f"Using lambda = {lambda_value}")
                        logger.info(f"Processing {repo.url}")
                        if start_with_pfr:
                            if "pfr" not in repo.url:
                                continue
                            else:
                                start_with_pfr = False
        
                        if single_repo:
                            repos_for_merged_dataset = []
                        
                        if not curriculum_learning:
                            result = add_repo_to_database(dynamic_database_json_path, repo, db)
                            if result is None:
                                write_skip_file(repo.url)
                                logger.info(f"Writing skip file for {repo.url}")

                        # Generate a new dataset from the dynamic database.
                        # The user can choose to generate a dataset from the entire dynamic database or a subset of it.
                        # TODO: don't always do merged_, if we change this then change the if condition in average test accordingly
                        dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_with_new_{dir_name}"
                        if (repo.url, repo.commit) not in repos_for_merged_dataset:
                            logger.info("Adding repo to repos_for_merged_dataset")
                            repos_for_merged_dataset.append((repo.url, repo.commit))
                        else:
                            logger.info("Repo already in repos_for_merged_dataset")
                        db.generate_merged_dataset(dst_dir, repos_for_merged_dataset)

                    # TODO: reduce repition later with all path
                    dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + f"merged_with_new_{dir_name}"
                    new_data_path = dst_dir

                    # All GPUs
                    logger.info("All GPUs")
                    model_checkpoint_path = None
                    best_model = None
                    data_module = None
                    if run_progressive_training:
                        try:
                            model_checkpoint_path = find_latest_checkpoint()
                            logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
                        except FileNotFoundError as e:
                            logger.error(str(e))
                            model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                        
                        # Train the model on the new dataset that we generated from the dynamic database.
                        logger.info("Inside train_test_fisher")
                        logger.info(f"Starting training at epoch {current_epoch}")
                        seed_everything(3407)

                        ### PROGRESSIVE TRAINING
                        
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

                        model = PremiseRetriever.load(
                            model_checkpoint_path, device, freeze=False, config=config
                        )
                        model.train()
                        logger.info(f"Loaded premise retriever at {model_checkpoint_path}")

                        # load previous Fisher Information Matrix for current EWC
                        if use_fisher:
                            latest_fisher = find_latest_fisher()
                            fisher_info = load_fisher_information(latest_fisher)
                            model.set_fisher_info(fisher_info)
                            logger.info("Fisher Information Matrix loaded.")

                        # TODO: use the yaml file instead of repeating here, same throughout
                        dir_name = new_data_path.split("/")[-1]
                        filename_suffix = f"_lambda_{lambda_value}"
                        checkpoint_callback = ModelCheckpoint(
                            dirpath=RAID_DIR + "/" + CHECKPOINT_DIR,
                            filename=dir_name + filename_suffix + "_{epoch}-{Recall@10_val:.2f}",
                            verbose=True,
                            save_top_k=-1,  # Save all checkpoints
                            every_n_epochs=1,  # Save every epoch (which is just once in this case)
                            monitor="Recall@10_val",
                            mode="max"
                        )
                        
                        early_stop_callback = EarlyStopping(
                            monitor="Recall@10_val",
                            patience=5,
                            mode="max",
                            verbose=True
                        )

                        lr_monitor = LearningRateMonitor(logging_interval='step')

                        # mid_epoch_checkpoint_dir = os.path.join(RAID_DIR, f"mid_epoch_checkpoints_{current_epoch}_{dir_name}_{use_fisher}_lambda_{lambda_value}")
                        # os.makedirs(mid_epoch_checkpoint_dir, exist_ok=True)
                        # timed_checkpoint_callback = TimedCheckpoint(checkpoint_dir=mid_epoch_checkpoint_dir)

                        VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60 * 52  # 1 year
                        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
                        os.environ['NCCL_TIMEOUT'] = str(VERY_LONG_TIMEOUT * 1000)

                        custom_log_dir = os.path.join(RAID_DIR, "lightning_logs", f"{dir_name}_{use_fisher}_lambda_{lambda_value}")
                        os.makedirs(custom_log_dir, exist_ok=True)

                        ddp_strategy = DDPStrategy(timeout=timedelta(seconds=VERY_LONG_TIMEOUT))
                        trainer = pl.Trainer(
                            accelerator="gpu",
                            gradient_clip_val=1.0,
                            precision="bf16-mixed",
                            strategy=ddp_strategy,
                            devices=4, # TODO: change for GPU
                            accumulate_grad_batches=4,
                            # callbacks=[lr_monitor, checkpoint_callback, early_stop_callback, timed_checkpoint_callback],
                            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
                            max_epochs=current_epoch + epochs_per_repo,
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            default_root_dir=custom_log_dir,
                        )

                        logger.info("right before barrier for data module")
                        trainer.strategy.barrier()
                        should_skip, skip_repo_url = should_skip_repo()
                        if should_skip:
                            logger.info(f"Skipping repository {skip_repo_url} due to preprocessing issues")
                            trainer.strategy.barrier()
                            if is_main_process:
                                logger.info("Removing skip file")
                                skip_file_path = os.path.join(RAID_DIR, DATA_DIR, "skip_repo.txt")
                                os.remove(skip_file_path)
                            continue

                        model.set_lambda(lambda_value)
                        corpus_path = new_data_path + "/corpus.jsonl"
                        data_path = new_data_path + "/random"
                        logger.info(f"Data path: {data_path}")
                        data_module = RetrievalDataModule(
                            data_path=data_path,
                            corpus_path=corpus_path,
                            num_negatives=3,
                            num_in_file_negatives=1,
                            model_name="google/byt5-small",
                            batch_size=BATCH_SIZE,
                            eval_batch_size=64,
                            max_seq_len=1024,
                            num_workers=4
                        )
                        data_module.setup(stage='fit')

                        logger.info(f"Training dataset size after load: {len(data_module.ds_train)}")
                        logger.info(f"Validation dataset size after load: {len(data_module.ds_val)}")
                        logger.info(f"Testing dataset size after load: {len(data_module.ds_pred)}")

                        logger.info(f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}")

                        try:
                            logger.info("hit the barrier before training")
                            trainer.strategy.barrier()
                            trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)
                            logger.info("hit the barrier after training")
                            trainer.strategy.barrier()
                        except Exception as e:
                            print(f"An error occurred during training: {str(e)}")
                            print(traceback.format_exc())

                        logger.info(f"Finished progressive training at epoch {trainer.current_epoch}")

                        ### TESTING FOR AVERAGE RECALL

                        # TODO: don't load corpus and reindex for every repo we use for average recall
                        try:
                            best_model_path = find_latest_checkpoint()
                            logger.info(f"Found latest checkpoint: {best_model_path}")
                            best_model = PremiseRetriever.load(best_model_path, device, freeze=False, config=config)
                        except FileNotFoundError as e:
                            logger.error(f"No checkpoint found: {str(e)}")
                            logger.warning("Using the current model state.")
                            best_model = model

                        best_model.eval()

                        logger.info("Testing...")
                        total_R1, total_R10, total_MRR = [], [], []
                        dataset_path = RAID_DIR + "/" + DATA_DIR
                        testing_paths = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]
                        if is_main_process:
                            with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                f.write("\n\n\n")
                                f.write(f"Results for {dir_name} with lambda = {lambda_value}")
                        for data_path in testing_paths:
                            # TODO: remove this for tests that do not use merged dataset
                            if "merged" not in data_path:
                                continue
                            # subprocess.run(["python","retrieval/main.py", "predict", "--config", "retrieval/confs/cli_lean4_random.yaml", "--ckpt_path", model_checkpoint_path, "--data-path", data_path], check=True)
                            run_cli(best_model_path, data_path)
                            if is_main_process:
                                num_gpus = 4 # TODO: change for GPU
                                preds_map = {}
                                for gpu_id in range(num_gpus):
                                    with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
                                        preds = pickle.load(f)
                                        preds_map.update(preds)

                                logger.info("Loaded the predictions pickle files")
                                data_path = os.path.join(data_path, "random", "test.json")
                                data = json.load(open(data_path))
                                logger.info(f"Evaluating on {data_path}")
                                R1, R10, MRR = _eval(data, preds_map)
                                logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
                                total_R1.append(R1)
                                total_R10.append(R10)
                                total_MRR.append(MRR)
                                with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                    f.write("\n\n\n")
                                    f.write(f"Intermediate results for {data_path}")
                                    f.write("\n\n\n")
                                    f.write(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

                        if is_main_process:
                            avg_R1 = np.mean(total_R1)
                            avg_R10 = np.mean(total_R10)
                            avg_MRR = np.mean(total_MRR)

                            logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

                            if not os.path.exists(EVAL_RESULTS_FILE_PATH):
                                open(EVAL_RESULTS_FILE_PATH, 'w').close()

                            with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                f.write("\n\n\n")
                                f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")
                    else:
                        model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                        if result is None:
                            logger.info(f"Skipping repository {repo.url} due to preprocessing issues")
                            continue

                    if is_main_process and run_progressive_training and use_fisher:
                        logger.info("Calculating Fisher Information Matrix for EWC")
                        ### FISHER INFORMATION MATRIX FOR NEXT EWC

                        # Switch to one GPU for calculating the Fisher Information Matrix
                        # TODO: barrier here
                        try:
                            # TODO: have separate intermediate checkpoints and save the epoch and data point, same for fisher
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            best_model.to(device)
                            train_dataloader = data_module.train_dataloader()
                            fisher_info = best_model.compute_fisher_information(train_dataloader, RAID_DIR + "/" + FISHER_DIR)
                            dir_path = RAID_DIR + "/" + FISHER_DIR
                            fisher_name = dir_path + "/" + dir_name + "_fisher_info.pkl"
                            with open(fisher_name, "wb") as f:
                                pickle.dump(fisher_info, f)
                            logger.info(f"Fisher info saved to {fisher_name}")
                        except Exception as e:
                            print(f"An error occurred during fisher: {str(e)}")
                            print(traceback.format_exc())

                    if is_main_process:
                        logger.info("Starting the prover")
                        # Set up the prover
                        use_vllm = False
                        corpus_path = dst_dir + "/corpus.jsonl"
                        tactic = None  # `None` since we are not using a fixed tactic generator
                        module = None  # `None` since we are not using a fixed tactic generator
                        num_workers = 4
                        num_gpus = 4 # TODO: change for GPU
                        timeout = 600
                        max_expansions = None
                        num_sampled_tactics = 64
                        debug = False
                        ckpt_path = f"{RAID_DIR}/model_lightning.ckpt"
                        prover = DistributedProver(
                            use_vllm,
                            ckpt_path,
                            corpus_path,
                            tactic,
                            module,
                            num_workers,
                            num_gpus=num_gpus,
                            timeout=timeout,
                            max_expansions=max_expansions,
                            num_sampled_tactics=num_sampled_tactics,
                            raid_dir=RAID_DIR,
                            checkpoint_dir=CHECKPOINT_DIR,
                            debug=debug,
                            run_progressive_training=run_progressive_training
                        )

                        # Prove sorry theorems
                        prove_sorry_theorems(db, prover, dynamic_database_json_path, repos_for_merged_dataset)
                        db.to_json(dynamic_database_json_path)

                        logger.info("Finished searching for proofs of sorry theorems")

                        # TODO: need to return proofs
                        # proofs = []
                        # Uncomment if you would like to contribute back to the repos!
                        # else:
                        #     base_branch = get_default_branch(repo_no_dir)
                        #     subprocess.run(["git", "-C", repo, "fetch", "origin", base_branch], check=True)
                        #     subprocess.run(["git", "-C", repo, "checkout", base_branch], check=True)
                        #     subprocess.run(["git", "-C", repo, "pull", "origin", base_branch], check=True)
                        #     create_or_switch_branch(repo, TMP_BRANCH, base_branch)
                        #     replace_sorry_with_proof(proofs)
                        #     committed = commit_changes(repo, COMMIT_MESSAGE)
                        #     if committed:
                        #         push_changes(repo, TMP_BRANCH)
                        #         url = str(create_pull_request(repo_no_dir, PR_TITLE, PR_BODY, TMP_BRANCH))
                        #         # TODO: add the PR URL to the database
                        #     shutil.rmtree(repo)
                    
                    logger.info("Finished processing the repository")
                    current_epoch += epochs_per_repo
                    logger.info(f"current epoch: {current_epoch}")
    except Exception as e:
        logger.info(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()
