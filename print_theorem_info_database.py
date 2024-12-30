from pathlib import Path
from dynamic_database import DynamicDatabase

def print_database_info(json_file_path: str):
    # Load the database from the JSON file
    db = DynamicDatabase.from_json(json_file_path)
    
    # Print information for each repository
    for repo in db.repositories:
        print(f"\nRepository: {repo.name}")
        print(f"URL: {repo.url}")
        print(f"Commit: {repo.commit}")
        print(f"Total Theorems: {repo.total_theorems}")
        print(f"Number of Premise Files: {repo.num_premise_files}")
        print(f"Number of Premises: {repo.num_premises}")
        print(f"Number of Files Traced: {repo.num_files_traced}")
        print("-" * 50)

if __name__ == "__main__":
    # Specify the path to your JSON file
    json_file_path = "../dynamic_database_PT_single_repo_no_ewc_curriculum_full_FINAL.json"
    
    # Call the function to print the database info
    print_database_info(json_file_path)