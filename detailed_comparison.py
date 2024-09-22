import sys
import json
from typing import List, Dict, Set
from dynamic_database import DynamicDatabase, Repository, Theorem

def load_database(file_path: str) -> DynamicDatabase:
    return DynamicDatabase.from_json(file_path)

def get_proved_sorry_theorems(repo: Repository) -> List[Theorem]:
    return repo.sorry_theorems_proved

def get_unproved_sorry_theorems(repo: Repository) -> List[Theorem]:
    return repo.sorry_theorems_unproved

def theorem_key(theorem: Theorem) -> tuple:
    return (theorem.full_name, str(theorem.file_path))

def print_theorem_details(theorem: Theorem, include_proof: bool = True):
    print(f"\nTheorem: {theorem.full_name}")
    print(f"File path: {theorem.file_path}")
    print(f"Theorem statement: {theorem.theorem_statement}")
    if include_proof and theorem.traced_tactics:
        print("Proof:")
        for tactic in theorem.traced_tactics:
            print(f"  {tactic.tactic}")
    print()

def compare_theorems(db1: DynamicDatabase, db2: DynamicDatabase, get_theorems_func, theorem_type: str):
    for repo1 in db1.repositories:
        repo2 = db2.get_repository(repo1.url, repo1.commit)
        if repo2 is None:
            print(f"Repository {repo1.url} (commit: {repo1.commit}) not found in the second database.")
            continue

        print(f"\n{'='*80}")
        print(f"Comparing {theorem_type} for repository: {repo1.url} (commit: {repo1.commit})")
        print(f"{'='*80}")

        theorems1 = get_theorems_func(repo1)
        theorems2 = get_theorems_func(repo2)

        theorems1_dict = {theorem_key(t): t for t in theorems1}
        theorems2_dict = {theorem_key(t): t for t in theorems2}

        only_in_db1 = set(theorems1_dict.keys()) - set(theorems2_dict.keys())
        only_in_db2 = set(theorems2_dict.keys()) - set(theorems1_dict.keys())
        common = set(theorems1_dict.keys()) & set(theorems2_dict.keys())

        print(f"\n{theorem_type} only in first database: {len(only_in_db1)}")
        for key in only_in_db1:
            print_theorem_details(theorems1_dict[key], include_proof=(theorem_type == "Proved sorry theorems"))

        print(f"\n{theorem_type} only in second database: {len(only_in_db2)}")
        for key in only_in_db2:
            print_theorem_details(theorems2_dict[key], include_proof=(theorem_type == "Proved sorry theorems"))

        print(f"\nCommon {theorem_type}: {len(common)}")
        for key in common:
            theorem1 = theorems1_dict[key]
            theorem2 = theorems2_dict[key]
            print_theorem_details(theorem1, include_proof=False)
            
            if theorem_type == "Proved sorry theorems":
                print("Proof:")
                if theorem1.traced_tactics:
                    for tactic in theorem1.traced_tactics:
                        print(f"  {tactic.tactic}")
                else:
                    print("  No detailed proof available.")
                
                proof1 = [t.tactic for t in theorem1.traced_tactics] if theorem1.traced_tactics else []
                proof2 = [t.tactic for t in theorem2.traced_tactics] if theorem2.traced_tactics else []
                
                if proof1 == proof2:
                    print("Proofs are identical in both databases.")
                else:
                    print("Proofs differ:")
                    print("Proof in second database:")
                    for tactic in proof2:
                        print(f"  {tactic}")
            print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <database1.json> <database2.json>")
        sys.exit(1)

    db1_path = sys.argv[1]
    db2_path = sys.argv[2]

    db1 = load_database(db1_path)
    db2 = load_database(db2_path)

    compare_theorems(db1, db2, get_proved_sorry_theorems, "Proved sorry theorems")
    compare_theorems(db1, db2, get_unproved_sorry_theorems, "Unproved sorry theorems")