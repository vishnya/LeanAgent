# from dynamic_database import DynamicDatabase
# from pathlib import Path
# from collections import defaultdict
# from typing import Dict, List
# import statistics

# def analyze_theorem_stats(database: DynamicDatabase):
#     results = []
    
#     for repo in database.repositories:
#         # Basic counts
#         stats = {
#             'repo_name': repo.name,
#             'repo_url': repo.url,
#             'commit': repo.commit,
#             'total_theorems': repo.total_theorems,
#             'proven_theorems': repo.num_proven_theorems,
#             'total_sorry_theorems': repo.num_sorry_theorems,
#             'sorry_proved': repo.num_sorry_theorems_proved,
#             'sorry_unproved': repo.num_sorry_theorems_unproved,
#             'total_premises': repo.num_premises,
#             'premise_files': repo.num_premise_files
#         }
        
#         # Analyze difficulty distributions
#         difficulty_stats = analyze_difficulty_distribution(repo)
#         stats.update(difficulty_stats)
        
#         # Get theorem counts by difficulty bracket
#         difficulty_brackets = get_difficulty_brackets(repo)
#         stats.update(difficulty_brackets)
        
#         results.append(stats)
    
#     return results

# def analyze_difficulty_distribution(repo):
#     difficulties = []
#     for theorem in repo.get_all_theorems:
#         if theorem.difficulty_rating is not None:
#             difficulties.append(theorem.difficulty_rating)
    
#     if not difficulties:
#         return {
#             'avg_difficulty': None,
#             'median_difficulty': None,
#             'min_difficulty': None,
#             'max_difficulty': None,
#             'std_dev_difficulty': None
#         }
    
#     return {
#         'avg_difficulty': statistics.mean(difficulties) if difficulties else None,
#         'median_difficulty': statistics.median(difficulties) if difficulties else None,
#         'min_difficulty': min(difficulties) if difficulties else None,
#         'max_difficulty': max(difficulties) if difficulties else None,
#         'std_dev_difficulty': statistics.stdev(difficulties) if len(difficulties) > 1 else None
#     }

# def get_difficulty_brackets(repo):
#     brackets = defaultdict(int)
    
#     for theorem in repo.get_all_theorems:
#         if theorem.difficulty_rating is not None:
#             if theorem.difficulty_rating < 1:
#                 brackets['very_easy'] += 1
#             elif theorem.difficulty_rating < 2:
#                 brackets['easy'] += 1
#             elif theorem.difficulty_rating < 3:
#                 brackets['medium'] += 1
#             elif theorem.difficulty_rating < 4:
#                 brackets['hard'] += 1
#             else:
#                 brackets['very_hard'] += 1
    
#     return {
#         'difficulty_distribution': dict(brackets)
#     }

# def print_repo_stats(results):
#     for repo in results:
#         print(f"\n{'='*80}")
#         print(f"Repository: {repo['repo_name']}")
#         print(f"URL: {repo['repo_url']}")
#         print(f"Commit: {repo['commit']}")
#         print(f"\nTheorem Counts:")
#         print(f"  Total Theorems: {repo['total_theorems']}")
#         print(f"  Proven Theorems: {repo['proven_theorems']}")
#         print(f"  Sorry Theorems: {repo['total_sorry_theorems']}")
#         print(f"    - Proved: {repo['sorry_proved']}")
#         print(f"    - Unproved: {repo['sorry_unproved']}")
#         print(f"\nPremise Information:")
#         print(f"  Total Premises: {repo['total_premises']}")
#         print(f"  Premise Files: {repo['premise_files']}")
        
#         if repo['avg_difficulty'] is not None:
#             print(f"\nDifficulty Statistics:")
#             print(f"  Average: {repo['avg_difficulty']:.2f}")
#             print(f"  Median: {repo['median_difficulty']:.2f}")
#             print(f"  Range: {repo['min_difficulty']:.2f} - {repo['max_difficulty']:.2f}")
#             if repo['std_dev_difficulty'] is not None:
#                 print(f"  Standard Deviation: {repo['std_dev_difficulty']:.2f}")
            
#             print("\nDifficulty Distribution:")
#             dist = repo['difficulty_distribution']
#             total = sum(dist.values())
#             for category, count in dist.items():
#                 percentage = (count / total * 100) if total > 0 else 0
#                 print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

# def main():
#     file_path = input("Enter the path to the JSON database file: ")
#     database = DynamicDatabase.from_json(file_path)
#     results = analyze_theorem_stats(database)
#     print_repo_stats(results)

# if __name__ == "__main__":
#     main()


# from dynamic_database import DynamicDatabase
# from pathlib import Path
# from collections import defaultdict
# from typing import Dict, List
# import statistics

# def analyze_theorem_stats(database: DynamicDatabase):
#     results = []
    
#     for repo in database.repositories:
#         # Get theorems that belong to this repo
#         repo_url = repo.url
#         repo_commit = repo.commit
        
#         own_theorems = [
#             thm for thm in (repo.proven_theorems + repo.sorry_theorems_proved + repo.sorry_theorems_unproved)
#             if thm.url == repo_url and thm.commit == repo_commit
#         ]
        
#         own_sorry_proved = [
#             thm for thm in own_theorems 
#             if thm in repo.sorry_theorems_proved
#         ]
        
#         own_sorry_unproved = [
#             thm for thm in own_theorems
#             if thm in repo.sorry_theorems_unproved
#         ]
        
#         own_proven = [
#             thm for thm in own_theorems
#             if thm in repo.proven_theorems
#         ]
        
#         # Basic counts using filtered theorems
#         stats = {
#             'repo_name': repo.name,
#             'repo_url': repo_url,
#             'commit': repo_commit,
#             'total_theorems': len(own_theorems),
#             'proven_theorems': len(own_proven),
#             'total_sorry_theorems': len(own_sorry_proved) + len(own_sorry_unproved),
#             'sorry_proved': len(own_sorry_proved),
#             'sorry_unproved': len(own_sorry_unproved),
#             'total_premises': repo.num_premises,
#             'premise_files': repo.num_premise_files
#         }
        
#         # Analyze difficulty distributions using filtered theorems
#         difficulty_stats = analyze_difficulty_distribution(own_theorems)
#         stats.update(difficulty_stats)
        
#         # Get theorem counts by difficulty bracket using filtered theorems
#         difficulty_brackets = get_difficulty_brackets(own_theorems)
#         stats.update(difficulty_brackets)
        
#         results.append(stats)
    
#     return results

# def analyze_difficulty_distribution(theorems):
#     difficulties = []
#     for theorem in theorems:
#         if theorem.difficulty_rating is not None:
#             difficulties.append(theorem.difficulty_rating)
    
#     if not difficulties:
#         return {
#             'avg_difficulty': None,
#             'median_difficulty': None,
#             'min_difficulty': None,
#             'max_difficulty': None,
#             'std_dev_difficulty': None
#         }
    
#     return {
#         'avg_difficulty': statistics.mean(difficulties) if difficulties else None,
#         'median_difficulty': statistics.median(difficulties) if difficulties else None,
#         'min_difficulty': min(difficulties) if difficulties else None,
#         'max_difficulty': max(difficulties) if difficulties else None,
#         'std_dev_difficulty': statistics.stdev(difficulties) if len(difficulties) > 1 else None
#     }

# def get_difficulty_brackets(theorems):
#     brackets = defaultdict(int)
    
#     for theorem in theorems:
#         if theorem.difficulty_rating is not None:
#             if theorem.difficulty_rating < 1:
#                 brackets['very_easy'] += 1
#             elif theorem.difficulty_rating < 2:
#                 brackets['easy'] += 1
#             elif theorem.difficulty_rating < 3:
#                 brackets['medium'] += 1
#             elif theorem.difficulty_rating < 4:
#                 brackets['hard'] += 1
#             else:
#                 brackets['very_hard'] += 1
    
#     return {
#         'difficulty_distribution': dict(brackets)
#     }

# def print_repo_stats(results):
#     for repo in results:
#         print(f"\n{'='*80}")
#         print(f"Repository: {repo['repo_name']}")
#         print(f"URL: {repo['repo_url']}")
#         print(f"Commit: {repo['commit']}")
#         print(f"\nTheorem Counts:")
#         print(f"  Total Theorems: {repo['total_theorems']}")
#         print(f"  Proven Theorems: {repo['proven_theorems']}")
#         print(f"  Sorry Theorems: {repo['total_sorry_theorems']}")
#         print(f"    - Proved: {repo['sorry_proved']}")
#         print(f"    - Unproved: {repo['sorry_unproved']}")
#         print(f"\nPremise Information:")
#         print(f"  Total Premises: {repo['total_premises']}")
#         print(f"  Premise Files: {repo['premise_files']}")
        
#         if repo['avg_difficulty'] is not None:
#             print(f"\nDifficulty Statistics:")
#             print(f"  Average: {repo['avg_difficulty']:.2f}")
#             print(f"  Median: {repo['median_difficulty']:.2f}")
#             print(f"  Range: {repo['min_difficulty']:.2f} - {repo['max_difficulty']:.2f}")
#             if repo['std_dev_difficulty'] is not None:
#                 print(f"  Standard Deviation: {repo['std_dev_difficulty']:.2f}")
            
#             print("\nDifficulty Distribution:")
#             dist = repo['difficulty_distribution']
#             total = sum(dist.values())
#             for category, count in dist.items():
#                 percentage = (count / total * 100) if total > 0 else 0
#                 print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

# def main():
#     file_path = input("Enter the path to the JSON database file: ")
#     database = DynamicDatabase.from_json(file_path)
#     results = analyze_theorem_stats(database)
#     print_repo_stats(results)

# if __name__ == "__main__":
#     main()


from dynamic_database import DynamicDatabase
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import statistics

def analyze_theorem_stats(database: DynamicDatabase):
    results = []
    
    for repo in database.repositories:
        # Get theorems that belong to this repo
        repo_url = repo.url
        repo_commit = repo.commit
        
        own_theorems = [
            thm for thm in (repo.proven_theorems + repo.sorry_theorems_proved + repo.sorry_theorems_unproved)
            if thm.url == repo_url and thm.commit == repo_commit
        ]
        
        own_sorry_proved = [
            thm for thm in own_theorems 
            if thm in repo.sorry_theorems_proved
        ]
        
        own_sorry_unproved = [
            thm for thm in own_theorems
            if thm in repo.sorry_theorems_unproved
        ]
        
        own_proven = [
            thm for thm in own_theorems
            if thm in repo.proven_theorems
        ]
        
        # Count premises that belong to this repo
        own_premises_count = 0
        own_premise_files = []
        for pf in repo.premise_files:
            premises_in_file = [
                p for p in pf.premises
                if any(thm.url == repo_url and thm.commit == repo_commit 
                      for thm in own_theorems 
                      if thm.file_path == pf.path)
            ]
            if premises_in_file:
                own_premises_count += len(premises_in_file)
                own_premise_files.append(pf)
        
        # Basic counts using filtered theorems and premises
        stats = {
            'repo_name': repo.name,
            'repo_url': repo_url,
            'commit': repo_commit,
            'total_theorems': len(own_theorems),
            'proven_theorems': len(own_proven),
            'total_sorry_theorems': len(own_sorry_proved) + len(own_sorry_unproved),
            'sorry_proved': len(own_sorry_proved),
            'sorry_unproved': len(own_sorry_unproved),
            'total_premises': own_premises_count,
            'premise_files': len(own_premise_files)
        }
        
        # Analyze difficulty distributions using filtered theorems
        difficulty_stats = analyze_difficulty_distribution(own_theorems)
        stats.update(difficulty_stats)
        
        # Get theorem counts by difficulty bracket using filtered theorems
        difficulty_brackets = get_difficulty_brackets(own_theorems)
        stats.update(difficulty_brackets)
        
        results.append(stats)
    
    return results

def analyze_difficulty_distribution(theorems):
    difficulties = []
    for theorem in theorems:
        if theorem.difficulty_rating is not None:
            difficulties.append(theorem.difficulty_rating)
    
    if not difficulties:
        return {
            'avg_difficulty': None,
            'median_difficulty': None,
            'min_difficulty': None,
            'max_difficulty': None,
            'std_dev_difficulty': None
        }
    
    return {
        'avg_difficulty': statistics.mean(difficulties) if difficulties else None,
        'median_difficulty': statistics.median(difficulties) if difficulties else None,
        'min_difficulty': min(difficulties) if difficulties else None,
        'max_difficulty': max(difficulties) if difficulties else None,
        'std_dev_difficulty': statistics.stdev(difficulties) if len(difficulties) > 1 else None
    }

def get_difficulty_brackets(theorems):
    brackets = defaultdict(int)
    
    for theorem in theorems:
        if theorem.difficulty_rating is not None:
            if theorem.difficulty_rating < 1:
                brackets['very_easy'] += 1
            elif theorem.difficulty_rating < 2:
                brackets['easy'] += 1
            elif theorem.difficulty_rating < 3:
                brackets['medium'] += 1
            elif theorem.difficulty_rating < 4:
                brackets['hard'] += 1
            else:
                brackets['very_hard'] += 1
    
    return {
        'difficulty_distribution': dict(brackets)
    }

def print_repo_stats(results):
    for repo in results:
        print(f"\n{'='*80}")
        print(f"Repository: {repo['repo_name']}")
        print(f"URL: {repo['repo_url']}")
        print(f"Commit: {repo['commit']}")
        print(f"\nTheorem Counts:")
        print(f"  Total Theorems: {repo['total_theorems']}")
        print(f"  Proven Theorems: {repo['proven_theorems']}")
        print(f"  Sorry Theorems: {repo['total_sorry_theorems']}")
        print(f"    - Proved: {repo['sorry_proved']}")
        print(f"    - Unproved: {repo['sorry_unproved']}")
        print(f"\nPremise Information:")
        print(f"  Total Premises: {repo['total_premises']}")
        print(f"  Premise Files: {repo['premise_files']}")
        
        if repo['avg_difficulty'] is not None:
            print(f"\nDifficulty Statistics:")
            print(f"  Average: {repo['avg_difficulty']:.2f}")
            print(f"  Median: {repo['median_difficulty']:.2f}")
            print(f"  Range: {repo['min_difficulty']:.2f} - {repo['max_difficulty']:.2f}")
            if repo['std_dev_difficulty'] is not None:
                print(f"  Standard Deviation: {repo['std_dev_difficulty']:.2f}")
            
            print("\nDifficulty Distribution:")
            dist = repo['difficulty_distribution']
            total = sum(dist.values())
            for category, count in dist.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

def main():
    file_path = input("Enter the path to the JSON database file: ")
    database = DynamicDatabase.from_json(file_path)
    results = analyze_theorem_stats(database)
    print_repo_stats(results)

if __name__ == "__main__":
    main()