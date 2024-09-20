import sys
import json
from typing import List, Dict, Set
from dynamic_database import DynamicDatabase, Repository, Theorem

def load_database(file_path: str) -> DynamicDatabase:
    return DynamicDatabase.from_json(file_path)

def get_proved_sorry_theorems(repo: Repository) -> List[Theorem]:
    return repo.sorry_theorems_proved

def theorem_key(theorem: Theorem) -> tuple:
    return (theorem.full_name, str(theorem.file_path))

def print_theorem_details(theorem: Theorem):
    print(f"\nTheorem: {theorem.full_name}")
    print(f"File path: {theorem.file_path}")
    print(f"Theorem statement: {theorem.theorem_statement}")
    print("Proof:")
    if theorem.traced_tactics:
        for tactic in theorem.traced_tactics:
            print(f"  {tactic.tactic}")
    else:
        print("  No detailed proof available.")
    print()

def compare_proved_sorry_theorems(db1: DynamicDatabase, db2: DynamicDatabase):
    for repo1 in db1.repositories:
        repo2 = db2.get_repository(repo1.url, repo1.commit)
        if repo2 is None:
            print(f"Repository {repo1.url} (commit: {repo1.commit}) not found in the second database.")
            continue

        print(f"\n{'='*80}")
        print(f"Comparing proved sorry theorems for repository: {repo1.url} (commit: {repo1.commit})")
        print(f"{'='*80}")

        proved_sorry1 = get_proved_sorry_theorems(repo1)
        proved_sorry2 = get_proved_sorry_theorems(repo2)

        proved_sorry1_set = set(theorem_key(t) for t in proved_sorry1)
        proved_sorry2_set = set(theorem_key(t) for t in proved_sorry2)

        only_in_db1 = proved_sorry1_set - proved_sorry2_set
        only_in_db2 = proved_sorry2_set - proved_sorry1_set
        common = proved_sorry1_set & proved_sorry2_set

        print(f"\nProved sorry theorems only in first database: {len(only_in_db1)}")
        for key in only_in_db1:
            theorem = next(t for t in proved_sorry1 if theorem_key(t) == key)
            print_theorem_details(theorem)

        print(f"\nProved sorry theorems only in second database: {len(only_in_db2)}")
        for key in only_in_db2:
            theorem = next(t for t in proved_sorry2 if theorem_key(t) == key)
            print_theorem_details(theorem)

        print(f"\nCommon proved sorry theorems: {len(common)}")
        for key in common:
            theorem = next(t for t in proved_sorry1 if theorem_key(t) == key)
            print_theorem_details(theorem)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <database1.json> <database2.json>")
        sys.exit(1)

    db1_path = sys.argv[1]
    db2_path = sys.argv[2]

    db1 = load_database(db1_path)
    db2 = load_database(db2_path)

    compare_proved_sorry_theorems(db1, db2)