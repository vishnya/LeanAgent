"""
This is the driver for LeanDojoBot. It will do the following things:
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

# TODO: here are some important features to implement (besides what's in the module docstring)
# 1. run this code every so often to check for updates to repos (host on a server)
# 2. if private, give user option to install

# TODO: add more as needed
known_repositories = [
    "leanprover/lean4",
    "leanprover-community/mathlib",
    "leanprover-community/mathlib4",
]

repos = []

personal_access_token = os.environ.get("PERSONAL_ACCESS_TOKEN")

def clone_repo(repo_url):
    print(f"Cloning {repo_url}")
    subprocess.run(["git", "clone", repo_url])

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
                repos.append(repo_full_name[repo_full_name.index("/") + 1:])  # remove owner from repo name
                clone_url = repo['clone_url']
                cloned_count += 1
                clone_repo(clone_url)
    else:
        print("Failed to search GitHub", response.status_code)

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
    # repo = LeanGitRepo(
    #     "https://github.com/teorth/pfr",
    #     "70d5c9f3f4602e6b3a1eaf263c55599213a73559",
    # )
    repo = LeanGitRepo(
        "https://github.com/Adarsh321123/lean4-example-adarsh",
        "f0cec352c953349d4b3885a05697f1c5a724892a",
    )
    logger.info(f"repo: {repo}")
    traced_repo = trace(repo, build_deps=False)
    data = []

    # TODO: do not trace the repos that are dependencies
    for thm in traced_repo.get_traced_theorems():
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
    # import ipdb; ipdb.set_trace()
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
    # import ipdb; ipdb.set_trace()

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
    # import ipdb; ipdb.set_trace()
    for result in results:
        if result.status == Status.PROVED:
            # logger.info(str(result))
            proof_text = "\n".join(result.proof)
            # TODO: find more efficient way to get url and repo name
            repo_name = result.theorem.repo.url.split("/")[-1]
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
    # TODO: remove the theorems that are not from the files in the repo itself

    # TODO: does this work with lemmas too?
    # evaluate.evaluate(data_path=data_path, ckpt_path=ckpt_path, split=split, num_workers=num_workers, num_gpus=num_gpus)

# def find_and_replace_sorry(directory):
#     # TODO: there are some edge cases where the sorry was used in a functional programming sense
#     # This can be seen sometimes in `LeanSMTParser/SMTParser/ParseSMTLemma.lean`
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".lean"):
#                 file_path = os.path.join(root, file)
#                 if ".lake" not in file_path:
#                     with open(file_path, 'r+') as f:
#                         content = f.read()
#                         if 'sorry' in content:
#                             print(f"found sorry in file {file_path}")
#                             retrieve_proof()
#                             # TODO: use this?
#                             # new_content = re.sub(r'\bsorry\b', 'proof_using_lean_copilot', content)
#                             # f.seek(0)
#                             # f.write(new_content)
#                             # f.truncate()

def replace_sorry_with_proof(proofs):
    # TODO: is this the right way to make a PR eventually?
    for proof in proofs:
        file_path, start, end, proof_text = proof
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        start_line, start_col = start.line_nb - 1, start.column_nb - 1
        end_line, end_col = end.line_nb - 1, end.column_nb - 1

        # Join the lines from start to end to form the text to be replaced
        original_text = ''.join(lines[start_line:end_line + 1])
        
        # Replace the `sorry` with the proof text
        new_text = original_text.replace('sorry', proof_text, 1)
        
        # Update the lines in the file
        lines[start_line:end_line + 1] = new_text.split('\n')
        
        with open(file_path, 'w') as file:
            file.writelines(lines)


def main():
    proofs = retrieve_proof() # TODO: uncomment
    # TODO: remove
    # proofs = [('lean4-example-adarsh/Lean4Example.lean', (3, 1), (4, 8), 'omega'), ('lean4-example-adarsh/Lean4Example.lean', (6, 1), (7, 8), 'simp')]
    import ipdb; ipdb.set_trace()
    replace_sorry_with_proof(proofs)
    # TODO: update hte control flow since we don't use find_and_replace_sorry anymore
    # TODO: uncomment
    # search_github_repositories(language="Lean", num_repos=5)
    # print(f"Found {len(repos)} repositories")
    # for repo in repos:
    #     # if repo != "BET": # TODO: remove
    #     #     continue
    #     print(f"Processing {repo}")
    #     find_and_replace_sorry(repo)
    #     # shutil.rmtree(repo)  # TODO: uncomment later for removing dir
    #     # TODO: make a branch too

if __name__ == "__main__":
    main()
