from dynamic_database import DynamicDatabase
from pathlib import Path

def analyze_sorry_theorems(database: DynamicDatabase):
    results = []
    for repo in database.repositories:
        repo_name = repo.name
        total_sorry = repo.num_sorry_theorems
        sorry_proved = repo.num_sorry_theorems_proved
        sorry_unproved = repo.num_sorry_theorems_unproved
        
        proved_theorems = [thm.full_name for thm in repo.sorry_theorems_proved]
        
        results.append({
            'repo_name': repo_name,
            'total_sorry': total_sorry,
            'sorry_proved': sorry_proved,
            'sorry_unproved': sorry_unproved,
            'proved_theorems': proved_theorems
        })
    
    return results

def print_stats(results):
    for repo in results:
        print(f"Repository: {repo['repo_name']}")
        print(f"  Total sorry theorems: {repo['total_sorry']}")
        print(f"  Proven sorry theorems: {repo['sorry_proved']}")
        print(f"  Unproven sorry theorems: {repo['sorry_unproved']}")
        print("  Proven sorry theorem names:")
        for thm in repo['proved_theorems']:
            print(f"    - {thm}")
        print()

def main():
    file_path = input("Enter the path to the JSON database file: ")
    database = DynamicDatabase.from_json(file_path)
    results = analyze_sorry_theorems(database)
    print_stats(results)

if __name__ == "__main__":
    main()