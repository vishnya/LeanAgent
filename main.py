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

from common import set_logger
from prover.proof_search import Status, DistributedProver

load_dotenv()

# TODO: add more as needed
known_repositories = [
    "leanprover/lean4",
    "leanprover-community/mathlib",
    "leanprover-community/mathlib4",
    "teorth/pfr",
]

repos = []

personal_access_token = os.environ.get("PERSONAL_ACCESS_TOKEN")

PR_TITLE = "[LeanCopilotBot] `sorry` Removed by Lean Copilot"

# TODO: make sure these links work when we release
PR_BODY = """We identify the files containing theorems that have `sorry`, and replace them with a proof discovered by [Lean Copilot](https://github.com/lean-dojo/LeanCopilot).

---

<i>~LeanCopilotBot - From the [LeanDojo](https://leandojo.org/) family</i>

[:octocat: repo](https://github.com/lean-dojo/LeanCopilotBot) | [🙋🏾 issues](https://github.com/lean-dojo/LeanCopilotBot/issues) | [🏪 marketplace](https://github.com/marketplace/LeanCopilotBot)
"""

# TODO: master?
TARGET_BRANCH = "main"

TMP_BRANCH = "_LeanCopilotBot"

COMMIT_MESSAGE = "[LeanCopilotBot] `sorry` Removed by Lean Copilot"

def clone_repo(repo_url):
    repo_name = "/".join(repo_url.split('/')[-2:]).replace('.git', '')
    print(f"Cloning {repo_url}")
    print(f"Repo name: {repo_name}")
    subprocess.run(["git", "clone", repo_url, repo_name])

def branch_exists(repo_name, branch_name):
    proc = subprocess.run(["git", "-C", repo_name, "branch", "-a"], capture_output=True, text=True)
    branches = proc.stdout.split('\n')
    local_branch = branch_name
    remote_branch = f'remote/{branch_name}'
    return any(branch.strip().endswith(local_branch) or branch.strip().endswith(remote_branch) for branch in branches)

def create_or_switch_branch(repo_name, branch_name):
    if not branch_exists(repo_name, branch_name):
        subprocess.run(["git", "-C", repo_name, "checkout", "-b", branch_name], check=True)
    else:
        subprocess.run(["git", "-C", repo_name, "checkout", branch_name], check=True)

def commit_changes(repo_name, commit_message):
    status = subprocess.run(["git", "-C", repo_name, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if status == "":
        print("No changes to commit.")
        return False
    subprocess.run(["git", "-C", repo_name, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_name, "commit", "-m", commit_message], check=True)
    return True

def push_changes(repo_name, branch_name):
    subprocess.run(["git", "-C", repo_name, "push", "-u", "origin", branch_name], check=True)

def create_pull_request(repo_full_name, title, body, head_branch, base_branch="main"):
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
    else:
        print("Failed to create pull request", response.text)

def search_github_repositories(language="Lean", num_repos=5):
    headers = {'Authorization': personal_access_token}
    query_params = {
        'q': f'language:{language}',
        'sort': 'updated',  # 'stars' may result in old repos
        'order': 'desc'
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
                repos.append(repo_full_name)
                clone_url = repo['clone_url']
                cloned_count += 1
                clone_repo(clone_url)
    else:
        print("Failed to search GitHub", response.status_code)

# TODO: remove
import re
_LEAN4_VERSION_REGEX = re.compile(r"leanprover/lean4:(?P<version>.+?)")
def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    return m["version"]

def is_supported_version(v) -> bool:
    """Check if ``v`` is at least `v4.3.0-rc2`."""
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3):
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
    else:
        return True

def change_toolchain_version(repo):
    # we need to change the toolchain version that the bot uses
    # to match the repo we are currently tracing
    # this will avoid lake build problems
    # Find the path to the desired Lean version using elan
    config = repo.get_config("lean-toolchain")
    logger.info(f"lean toolchain version: {config}")
    v = get_lean4_version_from_config(config["content"])
    logger.info(f"lean version v: {v}")
    logger.info(f"is supported: {is_supported_version(v)}")
    v = v[1:] # ignore v at beginning
    lean_dir = "/home/adarsh/.elan/toolchains/leanprover--lean4---" + v
    logger.info(f"lean path {lean_dir}")

    # # Extract the toolchain path from the full path to the Lean binary
    # lean_dir = os.path.dirname(os.path.dirname(lean_path))

    # Update LEAN4_PATH environment variable
    os.environ['LEAN4_PATH'] = lean_dir

    # Update PATH environment variable to include the Lean bin directory
    os.environ['PATH'] = f"{lean_dir}/bin:{os.environ.get('PATH', '')}"

    logger.info(f"Switched to Lean toolchain at: {lean_dir}")
    logger.info(f"lean --version: {subprocess.run(['lean', '--version'], capture_output=True).stdout.decode('utf-8')}")

def retrieve_proof():
    # data_path = "data/leandojo_benchmark_4/random/"
    ckpt_path = "kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_lightning.ckpt"
    indexed_corpus_path = None
    tactic = None
    module = None
    num_workers = 5
    num_gpus = 1
    timeout = 600
    num_sampled_tactics = 64
    verbose = False
    # TODO: change later
    # TODO: will this work with lakefile.toml
    # THIS WORKS on Lean (version 4.7.0-rc2, x86_64-unknown-linux-gnu, commit 6fce8f7d5cd1, Release)
    # repo = LeanGitRepo(
    #     "https://github.com/Adarsh321123/lean4-example-adarsh",
    #     "f0cec352c953349d4b3885a05697f1c5a724892a",
    # )
    # this requires leanprover/lean4:v4.8.0-rc1
    # repo = LeanGitRepo(
    #     "https://github.com/Adarsh321123/new-version-test",
    #     "279c3bc5c6d1e1b8810c99129d7d2c43c5469b54",
    # )
    # this requires leanprover/lean4:v4.8.0-rc2
    repo = LeanGitRepo(
        "https://github.com/Adarsh321123/new-new-version-test",
        "779fc7d7cc36755b76bda552118e910289ed3aa3",
    )
    # repo = LeanGitRepo(
    #     "https://github.com/teorth/pfr",
    #     "785a3d3cacc18889fdb9689cfc84edc97233886f",
    # )
    # lean_dir = "/home/adarsh/.elan/toolchains/leanprover--lean4---4.7.0"
    # os.environ['LEAN4_PATH'] = lean_dir
    # os.environ['PATH'] = f"{lean_dir}/bin:{os.environ.get('PATH', '')}"
    # logger.info(f"lean --version: {subprocess.run(['lean', '--version'], capture_output=True).stdout.decode('utf-8')}")
    change_toolchain_version(repo)
    logger.info(f"repo: {repo}")
    config = repo.get_config("lean-toolchain")
    logger.info(f"lean toolchain version: {config}")
    v = get_lean4_version_from_config(config["content"])
    logger.info(f"lean version v: {v}")
    logger.info(f"is supported: {is_supported_version(v)}")
    # traced_repo = trace(repo, build_deps=False)
    traced_repo = trace(repo)
    data = []

    # TODO: do not trace the repos that are dependencies???
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
        if (cur_repo == repo) and ("sorry" in cur_proof):  # avoid tracing Lean4 and other dependencies
            theorems.append(cur_theorem)
            positions.append(elem[2])
            ends.append(elem[3])

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
    proofs = []
    for result in results:
        if result.status == Status.PROVED:
            # logger.info(str(result))
            proof_text = "\n".join(result.proof)
            # TODO: find more efficient way to get url and repo name
            repo_name = "/".join(result.theorem.repo.url.split('/')[-2:]).replace('.git', '')
            file_path = repo_name + "/" + str(result.theorem.file_path)
            start = None
            end = None
            # TODO: optimize
            for i in range(len(theorems)):
                if theorems[i] == result.theorem:
                    start = positions[i]
                    end = ends[i]
            proofs.append((file_path, start, end, proof_text))

    return proofs

def replace_sorry_with_proof(proofs):
    logger.info(f"Replacing sorries with {len(proofs)} proofs!")
    # Group proofs by file paths
    proofs_by_file = {}
    for proof in proofs:
        file_path, start, end, proof_text = proof
        if file_path not in proofs_by_file:
            proofs_by_file[file_path] = []
        proofs_by_file[file_path].append((start, end, proof_text))
    
    # TODO: is this the right way to make a PR eventually?
    for file_path, proofs in proofs_by_file.items():
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # sort proof by starting line and column number
        proofs.sort(key=lambda x: (x[0].line_nb, x[0].column_nb), reverse=True)  # working bottom up retains positions
        
        for start, end, proof_text in proofs:
            start_line, start_col = start.line_nb - 1, start.column_nb - 1
            end_line, end_col = end.line_nb - 1, end.column_nb - 1

            # Join the lines from start to end to form the text to be replaced
            original_text = ''.join(lines[start_line:end_line + 1])
            
            # Replace the `sorry` with the proof text
            new_text = original_text.replace('sorry', proof_text, 1)
            
            # Update the lines in the file
            lines[start_line:end_line + 1] = new_text
            
            with open(file_path, 'w') as file:
                file.writelines(lines)


def main():
    search_github_repositories(language="Lean", num_repos=5)
    print(f"Found {len(repos)} repositories")
    # repos.append("Adarsh321123/new-version-test") # TODO: remove
    repos.append("Adarsh321123/new-new-version-test") # TODO: remove
    # repos.append("teorth/pfr") # TODO: remove
    for repo in repos:
        # if repo != "Adarsh321123/new-version-test": # TODO: remove
        #     continue
        if repo != "Adarsh321123/new-new-version-test": # TODO: remove
            continue
        # if repo != "teorth/pfr": # TODO: remove
        #     continue
        print(f"Processing {repo}")
        create_or_switch_branch(repo, TMP_BRANCH)
        proofs = retrieve_proof()
        replace_sorry_with_proof(proofs)
        import ipdb; ipdb.set_trace()
        committed = commit_changes(repo, COMMIT_MESSAGE)
        if committed:
            push_changes(repo, TMP_BRANCH)
            create_pull_request(repo, PR_TITLE, PR_BODY, TMP_BRANCH, TARGET_BRANCH)

        # shutil.rmtree(repo)  # TODO: uncomment later for removing dir, or actually we may need to keep?

if __name__ == "__main__":
    main()
