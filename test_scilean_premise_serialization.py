import os
import json
import re
from pathlib import Path
from tqdm import tqdm
from common import Premise

RAID_DIR = os.environ.get('RAID_DIR', '/workspace')
DATA_DIR = "datasets_PT_single_repo_no_ewc"

def escape_regex_special_chars(text):
    return re.escape(text)

def test_scilean_premise_serialization():
    # Find the SciLean dataset
    dataset_path = Path(RAID_DIR) / DATA_DIR
    scilean_dir = next(d for d in dataset_path.iterdir() if 'SciLean' in d.name)
    
    print(f"Testing SciLean dataset: {scilean_dir}")

    # Test premise serialization
    print("Testing premise serialization...")
    with open(str(scilean_dir / "corpus.jsonl"), 'r') as f:
        for line in tqdm(f):
            premise_data = json.loads(line)
            for p in premise_data['premises']:
                premise = Premise(
                    full_name=p['full_name'],
                    code=p['code'],
                    path=premise_data['path'],
                    start=p['start'],
                    end=p['end'],
                )
                try:
                    # Extract the prefix (assuming it's the part before the first dot)
                    prefix = premise.full_name.split('.')[0]
                    escaped_prefix = escape_regex_special_chars(prefix)
                    
                    # Construct the regex pattern
                    pattern = f"(?<=\\s)_?{escaped_prefix}_?"
                    
                    # Print debug information
                    print(f"Full name: {premise.full_name}")
                    print(f"Prefix: {prefix}")
                    print(f"Escaped prefix: {escaped_prefix}")
                    print(f"Regex pattern: {pattern}")
                    print(f"Pattern[9]: {pattern[9] if len(pattern) > 9 else 'N/A'}")
                    print(f"Code snippet: {premise.code[:50]}...")  # Print first 50 chars of code
                    
                    # Try to compile the regex
                    try:
                        re.compile(pattern)
                        print("Regex compiled successfully")
                    except re.error as regex_error:
                        print(f"Regex compilation error: {str(regex_error)}")
                    
                    # Try the substitution
                    new_code = re.sub(pattern, premise.full_name, premise.code)
                    print("Substitution successful")
                    print("-" * 50)
                except Exception as e:
                    print(f"Error processing premise: {premise.full_name}")
                    print(f"Error message: {str(e)}")
                    raise

    print("All premises processed!")

if __name__ == "__main__":
    test_scilean_premise_serialization()