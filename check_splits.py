import json
from pathlib import Path

def check_test_lean_in_splits(directory: Path):
    # Define the path to the JSON files
    train_path = directory / "train.json"
    valid_path = directory / "val.json"
    test_path = directory / "test.json"

    # Load the JSON data
    with train_path.open("r") as f:
        train_data = json.load(f)
    with valid_path.open("r") as f:
        valid_data = json.load(f)
    with test_path.open("r") as f:
        test_data = json.load(f)

    # Check for theorems from test.lean
    def contains_test_lean(theorems):
        return any("Test.lean" in thm["file_path"] for thm in theorems)

    # Verify the contents
    train_contains_test_lean = contains_test_lean(train_data)
    valid_contains_test_lean = contains_test_lean(valid_data)
    test_contains_test_lean = contains_test_lean(test_data)

    # Print results
    print(f"Train contains test.lean theorems: {train_contains_test_lean}")
    print(f"Valid contains test.lean theorems: {valid_contains_test_lean}")
    print(f"Test contains test.lean theorems: {test_contains_test_lean}")

    # Check for theorems from test.lean and print their names
    def print_test_lean_theorems(theorems, split_name):
        test_lean_theorems = [thm["file_path"] for thm in theorems if "Test.lean" in thm["file_path"]]
        if test_lean_theorems:
            print(f"{split_name} contains test.lean theorems:")
            for name in test_lean_theorems:
                print(f"  - {name}")
        else:
            print(f"{split_name} does not contain any test.lean theorems.")

    # Verify the contents
    print_test_lean_theorems(train_data, "Train")
    print_test_lean_theorems(valid_data, "Valid")
    print_test_lean_theorems(test_data, "Test")

if __name__ == "__main__":
    # Set the directory path
    directory = Path("~/lean_project/datasets_PT_single_repo_no_ewc_curriculum_minif2f_test/miniF2F-lean4_9e445f5435407f014b88b44a98436d50dd7abd00/random").expanduser()
    check_test_lean_in_splits(directory)