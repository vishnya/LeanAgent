"""
This is the driver for LeanBot. It will do the following things:
1. Search for repositories with Lean files in them.
2. Clone these repositories.
3. Find theorems with `sorry` in them and replace them with a proof.
4. Commit and push the changes.
5. Open a pull request.
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
import numpy as np
import hashlib
import argparse
from tqdm import tqdm
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
import generate_benchmark_lean4
import traceback
import sys
from tqdm import tqdm

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
from retrieval.main import run_cli
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything

# TODO: standardize with all Path or just string
random.seed(3407)  # https://arxiv.org/abs/2109.08203
# TODO: constant?
# TODO: do we still need repo_dir
repo_dir = "/raid/adarsh/repos_new" # TODO: for release change these back to <DIR>
RAID_DIR = "/raid/adarsh"
DATA_DIR = "datasets_new"
CHECKPOINT_DIR = "checkpoints_new"
FISHER_DIR = "fisher_new"
# TODO: do we still need this?
load_dotenv()

global_results = {
    "total_repositories": 0,
    "repositories": {}
}

# Feel free to remove any repos from this list if you would like to test on them
known_repositories = [
    "leanprover-community/mathlib4",  # ReProver is trained on this + LeanDojo already tests on it
    "leanprover-community/batteries", # ReProver is trained on this + LeanDojo already tests on it
    "leanprover-community/aesop",
    "leanprover/lean4",
    "leanprover-community/mathlib",
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
    "kmill/lean4-raytracer",
    "argumentcomputer/yatima",
    "ImperialCollegeLondon/formalising-mathematics-2024",
    "leanprover-community/ProofWidgets4",
    "leanprover/verso",
    "leanprover-community/NNG4",
    "ufmg-smite/lean-smt",
    "google-deepmind/debate",
    "teorth/symmetric_project",
    "cmu-l3/llmlean",
    "PatrickMassot/GlimpseOfLean",
    "avigad/lamr",
    "leanprover-community/quote4",
    "yuma-mizuno/lean-math-workshop",
    "leanprover-community/iris-lean",
    "aripiprazole/rinha",
    "loganrjmurphy/LeanEuclid",
    "leanprover/lean4-cli",
    "leanprover/LeanInk",
    "leanprover-community/lean-auto",
    "leanprover-community/repl",
    "leanprover/doc-gen4",
    "leanprover-community/con-nf",
    "FormalizedFormalLogic/Foundation",
    "leanprover/SampCert",
    "nomeata/loogle",
    "risc0/risc0-lean4",
    "siddhartha-gadgil/Saturn",
    "leanprover-community/flt-regular",
    "eric-wieser/lean-matrix-cookbook",
    "PatrickMassot/verbose-lean4",
    "tydeu/lean4-alloy",
    "opencompl/lean-mlir-old",
    "leanprover/leansat",
    "BoltonBailey/formal-snarks-project",
    "dwrensha/lean4-maze",
    "leanprover/LNSym",
    "leanprover-community/mathport",
    "forked-from-1kasper/ground_zero",
    "mo271/formal_book",
    "rami3l/plfl",
]

repos = []  # stores the names of all the repos
lean_git_repos = []  # stores the LeanGitRepo objects

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
            logger.info(f"Processing {repo_full_name}")
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
                    cloned_count += 1 # TODO: only increase if compatible commit found
                except Exception as e:
                    # shutil.rmtree(name)
                    # Note: some 404s happen since some repos don't have a lean-toolchain
                    # but still use Lean
                    logger.info(f"Failed to clone {repo_full_name} because of {e}")
            else:
                logger.info(f"Skipping {repo_full_name} since it is a known repository")
    else:
        logger.info("Failed to search GitHub", response.status_code)


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

def train_test_fisher(model_checkpoint_path, new_data_path, lambda_value, current_epoch, epochs_per_repo=1):
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
    logger.info(f"Loaded premise retriever at {model_checkpoint_path} and reset epoch count")

    # load previous Fisher Information Matrix for current EWC
    latest_fisher = find_latest_fisher()
    fisher_info = load_fisher_information(latest_fisher)
    model.set_fisher_info(fisher_info)
    logger.info("Fisher Information Matrix loaded.")

    # TODO: try with novel split later instead of random
    # TODO: use the yaml file instead of repeating here, same throughout
    model.set_lambda(lambda_value)
    corpus_path = new_data_path + "/corpus.jsonl"
    data_path = new_data_path + "/random"
    print(f"Data path: {data_path}")
    data_module = RetrievalDataModule(
        data_path=data_path,
        corpus_path=corpus_path,
        num_negatives=3,
        num_in_file_negatives=1,
        model_name="google/byt5-small",
        batch_size=4,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=4
    )
    data_module.setup(stage='fit')

    dir_name = new_data_path.split("/")[-1]
    filename_suffix = f"_lambda_{lambda_value}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=RAID_DIR + "/" + CHECKPOINT_DIR,
        filename=dir_name + filename_suffix + "_{epoch}-{Recall@10_val:.2f}",
        verbose=True,
        save_top_k=1,
        save_last=False,
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

    print(f"Training dataset size after load: {len(data_module.ds_train)}")
    print(f"Validation dataset size after load: {len(data_module.ds_val)}")
    print(f"Testing dataset size after load: {len(data_module.ds_pred)}")

    trainer = pl.Trainer(
        accelerator="gpu",
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        strategy="ddp",
        devices=1, # TODO: change for GPU
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
        max_epochs=current_epoch + epochs_per_repo,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=os.path.join("lightning_logs_main")  # TODO: can remove later
    )

    logger.info(f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}")

    trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)

    logger.info(f"Finished progressive training at epoch {trainer.current_epoch}")

    ### TESTING FOR AVERAGE RECALL

    # TODO: uncomment later
    # TODO: don't load corpus and reindex for every repo we use for average recall
    # 
    # # Load the best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model = PremiseRetriever.load(best_model_path, device, freeze=False, config=config)
    else:
        logger.warning("No best model found. Using the last trained model.")
        best_model = model
    best_model.eval()

    logger.info("Testing...")
    total_R1, total_R10, total_MRR = [], [], []
    dataset_path = RAID_DIR + "/" + DATA_DIR
    testing_paths = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]
    for data_path in testing_paths:
        # subprocess.run(["python","retrieval/main.py", "predict", "--config", "retrieval/confs/cli_lean4_random.yaml", "--ckpt_path", model_checkpoint_path, "--data-path", data_path], check=True)
        run_cli(best_model_path, data_path)
        num_gpus = 1 # TODO: change for GPU
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

    avg_R1 = np.mean(total_R1)
    avg_R10 = np.mean(total_R10)
    avg_MRR = np.mean(total_MRR)

    logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

    # Save average accuracies to a file
    file_path = "/home/adarsh/ReProver/total_evaluation_results_full_PT.txt"
    with open(file_path, "a") as f:
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write(f"Results for {dir_name} with lambda = {lambda_value}")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

    
    ### FISHER INFORMATION MATRIX FOR NEXT EWC

    # Switch to one GPU for calculating the Fisher Information Matrix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    train_dataloader = data_module.train_dataloader()
    fisher_info = best_model.compute_fisher_information(train_dataloader)
    dir_path = RAID_DIR + "/" + FISHER_DIR
    fisher_name = dir_path + "/" + dir_name + "_fisher_info.pkl"
    with open(fisher_name, "wb") as f:
        pickle.dump(fisher_info, f)
    logger.info(f"Fisher info saved to {fisher_name}")

    # TODO: add anything else from yaml conf if needed

    return model

def update_and_output_results(repo_url: str, repository_results, lambda_value: float):
    global global_results
    global_results["repositories"][repo_url] = repository_results
    results_file = f"results_lambda_{lambda_value}_PT_matrix_cookbook_partial.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(global_results, f, indent=4, ensure_ascii=False)
    logger.info(f"Partial results saved to {results_file}")

def retrieve_proof(repo, repo_no_dir, sha, lambda_value, current_epoch, epochs_per_repo):
    # TODO: update comments throughout
    """
    This method does the following:
    1. Check if the given repo is supported.
    2. Trace the repo.
    3. Generate a corpus of the repo's premises.
    4. Search for proofs for theorems with `sorry` in them.
    """
    if ray.is_initialized():
        ray.shutdown()

    url = repo.url
    if not url.endswith('.git'):
        url = url + '.git'

    logger.info(f"Processing {url}")
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
    traced_repo, num_premises, num_files_traced = generate_benchmark_lean4.main(repo.url, sha, dst_dir)
    if not traced_repo:
        logger.info(f"Failed to trace {url}")
        return None
    logger.info(f"Finished generating benchmark at {dst_dir}")

    # model_checkpoint_path = "/raid/adarsh/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
    try:
        model_checkpoint_path = find_latest_checkpoint()
        logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    
    train_test_fisher(model_checkpoint_path, dst_dir, lambda_value, current_epoch, epochs_per_repo)

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

    logger.info("Traced all theorems")

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

    corpus_path = dst_dir + "/corpus.jsonl"
    tactic = None  # `None` since we are not using a fixed tactic generator
    module = None  # `None` since we are not using a fixed tactic generator
    num_workers = 1 # TODO: do everywhere if good
    num_gpus = 1 # TODO: change for GPU
    timeout = 600
    num_sampled_tactics = 64
    debug = False
    ckpt_path = "/raid/adarsh/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_lightning.ckpt"
    logger.info("About to search for proofs")
    prover = DistributedProver(
        ckpt_path,
        corpus_path,
        tactic,
        module,
        num_workers,
        num_gpus=num_gpus,
        timeout=timeout,
        num_sampled_tactics=num_sampled_tactics,
        debug=debug,
    )

    proofs = []
    num_sorries = len(theorems)
    logger.info(f"Found {num_sorries} sorries!")
    repository_results = {
        "commit": sha,
        "number_of_sorries": num_sorries,
        "number_of_proofs_found": 0,
        "proofs_details": [],
        "unproved_sorries": [],
        "number_of_premises_theorems_retrieved": num_premises,
        "num_files_traced": num_files_traced,
        "PR": "",
    }
    for i in tqdm(range(num_sorries), desc="Processing theorems", unit="theorem"):
        theorem = theorems[i]
        position = positions[i]
        logger.info(f"Searching for proof for {theorem.full_name}")
        logger.info(f"Position: {position}")

        results = prover.search_unordered(repo, [theorem], [position])
        result = results[0] if results else None
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
                repository_results["number_of_proofs_found"] += 1
                repository_results["proofs_details"].append({
                    "file_path": file_path,
                    "theorem_name": theorem_name,
                    "proof_text": proof_text
                })
                proofs.append((file_path, start, end, proof_text, theorem_name))
            else:
                repo_name = "/".join(result.theorem.repo.url.split('/')[-2:]).replace('.git', '')
                repo_name = repo_dir + "/" + repo_name
                file_path = repo_name + "/" + str(result.theorem.file_path)
                theorem_name = str(result.theorem.full_name)
                repository_results["unproved_sorries"].append({
                    "file_path": file_path,
                    "theorem_name": theorem_name
                })
            
            # Update global results and output partial results after each proof is processed
            update_and_output_results(repo.url, repository_results, lambda_value)
    
    logger.info("Finished searching for proofs")

    return repository_results, proofs

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

def main():
    """The main function that drives the bot."""
    try:
        # TODO: incorporate latest changes from ReProver repo
        # TODO: for PR, we need to make fork and then push to there
        # lambdas = [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000]
        # lambdas = [0.01, 0.1, 1, 10, 100, 10000]
        # lambdas = [1, 10, 100, 10000]
        lambdas = [0.0]
        # lambdas = [0.01]
        # lambdas = [0.0]

        logger.info("Configuring LeanDojo...")
        generate_benchmark_lean4.configure_leandojo()
        logger.info("LeanDojo configured")

        clone_url = "https://github.com/eric-wieser/lean-matrix-cookbook.git"
        repo_name, sha = clone_repo(clone_url)
        url = clone_url.replace('.git', '')
        lean_git_repo = LeanGitRepo(url, sha)
        lean_git_repos.append(lean_git_repo)
        repos.append("eric-wieser/lean-matrix-cookbook")

        current_epoch = 1 # TODO: change
        epochs_per_repo = 1

        # search_github_repositories("Lean", 30)
        num_repos = len(repos)
        global_results["total_repositories"] = num_repos
        print(f"Found {num_repos} repositories")

        for i in range(num_repos):
            for lambda_value in lambdas:
                print(f"Training and testing with lambda = {lambda_value}")
                repo = repos[i]
                repo_no_dir = repo
                repo = repo_dir + "/" + repo
                lean_git_repo = lean_git_repos[i]
                print(f"Processing {repo}")
                repository_results, proofs = retrieve_proof(lean_git_repo, repo_no_dir, lean_git_repo.commit, lambda_value, current_epoch, epochs_per_repo)
                current_epoch += epochs_per_repo
                if repository_results is None:
                    logger.info("Skipping repository due to configuration or error.")
                    continue
                
                if proofs:
                    base_branch = get_default_branch(repo_no_dir)
                    subprocess.run(["git", "-C", repo, "fetch", "origin", base_branch], check=True)
                    subprocess.run(["git", "-C", repo, "checkout", base_branch], check=True)
                    subprocess.run(["git", "-C", repo, "pull", "origin", base_branch], check=True)
                    create_or_switch_branch(repo, TMP_BRANCH, base_branch)
                    # Uncomment if you would like to contribute back to the repos!
                    # replace_sorry_with_proof(proofs)
                    # committed = commit_changes(repo, COMMIT_MESSAGE)
                    # if committed:
                    #     push_changes(repo, TMP_BRANCH)
                    #     url = str(create_pull_request(repo_no_dir, PR_TITLE, PR_BODY, TMP_BRANCH))
                    #     global_results["repositories"][lean_git_repo.url]["PR"] = url
                    # shutil.rmtree(repo)

                results_file = f"results_lambda_{lambda_value}_PT_matrix_cookbook.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(global_results, f, indent=4, ensure_ascii=False)
                    logger.info(f"Final results saved to {results_file}")
    except Exception as e:
        logger.info(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # Ensure final results are saved even if an exception occurs
        results_file = f"results_lambda_{lambda_value}_PT_matrix_cookbook.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(global_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Exception occurred. Final results saved to {results_file}")

if __name__ == "__main__":
    main()
