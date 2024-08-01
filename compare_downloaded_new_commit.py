import json
from pathlib import Path

downloaded_dataset_path = 'leandojo_benchmark_4_downloaded/random'
new_commit_dataset_path = 'leandojo_benchmark_4_new_commit/random'
downloaded_corpus_path = 'leandojo_benchmark_4_downloaded/corpus.jsonl'
new_commit_corpus_path = 'leandojo_benchmark_4_new_commit/corpus.jsonl'

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        print(f"Loading {file_path}")
        return [json.loads(line) for line in file]

def load_json(file_path):
    datasets = {}
    for split in ["train", "val", "test"]:
        new_file_path = f"{file_path}/{split}.json"
        with open(new_file_path, 'r') as file:
            print(f"Loading {new_file_path}")
            datasets[split] = json.load(file)
    return datasets

downloaded_dataset = load_json(downloaded_dataset_path)
downloaded_dataset_train, downloaded_dataset_val, downloaded_dataset_test = downloaded_dataset["train"], downloaded_dataset["val"], downloaded_dataset["test"]
new_commit_dataset = load_json(new_commit_dataset_path)
new_commit_dataset_train, new_commit_dataset_val, new_commit_dataset_test = new_commit_dataset["train"], new_commit_dataset["val"], new_commit_dataset["test"]
downloaded_corpus = load_jsonl(downloaded_corpus_path)
new_commit_corpus = load_jsonl(new_commit_corpus_path)

analysis_results = {
    "downloaded_dataset_train_size": len(downloaded_dataset_train),
    "downloaded_dataset_val_size": len(downloaded_dataset_val),
    "downloaded_dataset_test_size": len(downloaded_dataset_test),
    "new_commit_dataset_train_size": len(new_commit_dataset_train),
    "new_commit_dataset_val_size": len(new_commit_dataset_val),
    "new_commit_dataset_test_size": len(new_commit_dataset_test),
    "downloaded_corpus_size": len(downloaded_corpus),
    "new_commit_corpus_size": len(new_commit_corpus),
    "downloaded_corpus_file_premises_size": len(downloaded_corpus[0]['premises']),
    "new_commit_corpus_file_premises_size": len(new_commit_corpus[0]['premises']),
}

print(analysis_results)
