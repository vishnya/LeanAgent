import json
import os

def check_duplicates(file_path):
    if file_path.endswith('.jsonl'):
        # For corpus.jsonl
        unique_paths = set()
        duplicates = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                line_json = json.loads(line)
                path = line_json["path"]
                if path in unique_paths:
                    duplicates.append(i)
                else:
                    unique_paths.add(path)
        return len(unique_paths), len(duplicates), duplicates
    else:
        # For JSON files (train.json, val.json, test.json)
        with open(file_path, 'r') as f:
            data = json.load(f)
        unique_items = set()
        duplicates = []
        for i, item in enumerate(data, 1):
            item_tuple = (item['file_path'], item['full_name'], item['start'][0], item['start'][1], item['end'][0], item['end'][1])
            if item_tuple in unique_items:
                duplicates.append(i)
            else:
                unique_items.add(item_tuple)
        return len(unique_items), len(duplicates), duplicates

def verify_no_duplicates(merged_dir, old_merged_dir=None):
    files_to_check = ['corpus.jsonl', 'random/train.json', 'random/val.json', 'random/test.json',
                      'novel_premises/train.json', 'novel_premises/val.json', 'novel_premises/test.json']

    has_duplicates = False

    for file_name in files_to_check:
        file_path = os.path.join(merged_dir, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        unique_count, duplicate_count, _ = check_duplicates(file_path)
        print(f"File: {file_name}")
        print(f"  Unique items: {unique_count}")
        print(f"  Duplicates: {duplicate_count}")
        if duplicate_count > 0:
            has_duplicates = True

        if old_merged_dir:
            old_file_path = os.path.join(old_merged_dir, file_name)
            if os.path.exists(old_file_path):
                old_unique_count, old_duplicate_count, _ = check_duplicates(old_file_path)
                print(f"  Previous unique items: {old_unique_count}")
                print(f"  Previous duplicates: {old_duplicate_count}")
                print(f"  Difference in unique items: {unique_count - old_unique_count}")
            else:
                print(f"  Previous file not found: {old_file_path}")
        
        print()

    return has_duplicates

if __name__ == "__main__":
    # Usage
    ROOT_DIR = "/raid/adarsh"
    DATA_DIR = "datasets_test"
    MERGED_DATA_DIR = "datasets_merged"
    merged_dir = os.path.join(ROOT_DIR, MERGED_DATA_DIR, "merged_pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5_generated")
    old_merged_dir = os.path.join(ROOT_DIR, MERGED_DATA_DIR, "merged_pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5_updated")

    verify_no_duplicates(merged_dir, old_merged_dir)