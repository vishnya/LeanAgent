from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from dynamic_database import DynamicDatabase, Repository, Theorem

@dataclass(frozen=True)
class TheoremInfo:
    full_name: str
    file_path: str
    url: str
    commit: str

def get_proven_sorry_theorems(database: DynamicDatabase) -> Dict[Tuple[str, str], Set[TheoremInfo]]:
    proven_sorry_theorems = {}
    for repo in database.repositories:
        repo_key = (repo.url, repo.commit)
        proven_sorry_theorems[repo_key] = set()
        for theorem in repo.sorry_theorems_proved:
            theorem_info = TheoremInfo(
                full_name=theorem.full_name,
                file_path=str(theorem.file_path),
                url=repo.url,
                commit=repo.commit
            )
            proven_sorry_theorems[repo_key].add(theorem_info)
    return proven_sorry_theorems

def compare_databases(db1: DynamicDatabase, db2: DynamicDatabase) -> Dict:
    proven_sorry_1 = get_proven_sorry_theorems(db1)
    proven_sorry_2 = get_proven_sorry_theorems(db2)

    all_repos = set(proven_sorry_1.keys()) | set(proven_sorry_2.keys())

    comparison = {
        'common_repos': [],
        'only_in_db1': [],
        'only_in_db2': [],
        'repo_comparisons': {}
    }

    for repo_key in all_repos:
        if repo_key in proven_sorry_1 and repo_key in proven_sorry_2:
            comparison['common_repos'].append(repo_key)
            theorems_1 = proven_sorry_1[repo_key]
            theorems_2 = proven_sorry_2[repo_key]

            common_theorems = theorems_1 & theorems_2
            only_in_1 = theorems_1 - theorems_2
            only_in_2 = theorems_2 - theorems_1

            comparison['repo_comparisons'][repo_key] = {
                'common_theorems': [t.__dict__ for t in common_theorems],
                'only_in_db1': [t.__dict__ for t in only_in_1],
                'only_in_db2': [t.__dict__ for t in only_in_2]
            }
        elif repo_key in proven_sorry_1:
            comparison['only_in_db1'].append(repo_key)
        else:
            comparison['only_in_db2'].append(repo_key)

    return comparison

def print_comparison(comparison: Dict):
    print("Database Comparison Report")
    print("==========================")
    
    print("\nCommon Repositories:")
    for repo in comparison['common_repos']:
        print(f"- {repo[0]} (commit: {repo[1]})")
    
    print("\nRepositories only in Database 1:")
    for repo in comparison['only_in_db1']:
        print(f"- {repo[0]} (commit: {repo[1]})")
    
    print("\nRepositories only in Database 2:")
    for repo in comparison['only_in_db2']:
        print(f"- {repo[0]} (commit: {repo[1]})")
    
    print("\nDetailed Comparison of Common Repositories:")
    for repo, repo_comparison in comparison['repo_comparisons'].items():
        print(f"\nRepository: {repo[0]} (commit: {repo[1]})")
        
        print("  Theorems proven in both databases:")
        for theorem in repo_comparison['common_theorems']:
            print(f"  - {theorem['full_name']} (file: {theorem['file_path']})")
        
        print("  Theorems only proven in Database 1:")
        for theorem in repo_comparison['only_in_db1']:
            print(f"  - {theorem['full_name']} (file: {theorem['file_path']})")
        
        print("  Theorems only proven in Database 2:")
        for theorem in repo_comparison['only_in_db2']:
            print(f"  - {theorem['full_name']} (file: {theorem['file_path']})")

def main(db1_path: str, db2_path: str):
    db1 = DynamicDatabase.from_json(db1_path)
    db2 = DynamicDatabase.from_json(db2_path)
    
    comparison = compare_databases(db1, db2)
    print_comparison(comparison)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_db1.json> <path_to_db2.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])