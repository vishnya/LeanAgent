import ray
from lean_dojo import LeanGitRepo
import subprocess
import re
import generate_benchmark_lean4
import scripts.convert_t5encoder_to_ct2 as convert_t5encoder_to_ct2
import os
import json
import torch
import requests
from huggingface_hub import hf_hub_download, HfApi, upload_file, upload_folder
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything
from retrieval.datamodule import RetrievalDataModule
from retrieval.model import PremiseRetriever
import pytorch_lightning as pl
import pickle
from common import IndexedCorpus
import numpy as np

ROOT_DIR = os.environ.get('ROOT_DIR', '/workspace')
DATA_DIR = os.environ.get('DATA_DIR', 'datasets')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
HUGGINGFACE_API_URL = os.environ.get('HUGGINGFACE_API_URL', 'https://huggingface.co/api/models')
USER = os.environ.get('USER', 'AK123321')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

def merge_datasets():
    data_dir = ROOT_DIR + "/" + DATA_DIR
    merged_dir = data_dir + "/" + "merged"
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    split_strategies = ['random', 'novel_premises']
    split_types = ['train', 'val', 'test']

    for strategy in split_strategies:
        print(f"Merging datasets for {strategy}")
        strategy_dir = merged_dir + "/" + strategy
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)

        for split in split_types:
            print(f"Processing {split} split")
            merged_data = []
            
            for dataset in os.listdir(data_dir):
                print(f"Processing {dataset}")
                dataset_path = os.path.join(data_dir, dataset)
                if os.path.isdir(dataset_path):
                    json_file = os.path.join(dataset_path, strategy, f"{split}.json")
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            merged_data.extend(data)
            
            output_file = os.path.join(strategy_dir, f"{split}.json")
            with open(output_file, 'w') as f:
                json.dump(merged_data, f)
            print(f"Finished processing {split} split")
        print(f"Finished merging datasets for {strategy}")
    
    merged_corpus = []
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path):
            corpus_file = os.path.join(dataset_path, "corpus.jsonl")
            if os.path.exists(corpus_file):
                with open(corpus_file, 'r') as f:
                    for line in f:
                        merged_corpus.append(json.loads(line))
    
    with open(os.path.join(merged_dir, "corpus.jsonl"), 'w') as f:
        for item in merged_corpus:
            f.write(json.dumps(item) + '\n')

    print("Deleting individual datasets")
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path) and dataset != "merged":
            print(f"Deleting dataset: {dataset}")
            shutil.rmtree(dataset_path)

def get_compatible_commit(url):
    # TODO: optimize with binary search
    # TODO: maybe faster to get toolchiain version wihtout making git repo eeach time
    try:
        process = subprocess.Popen(["git", "ls-remote", url], stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        latest_commit = re.split(r'\t+', stdout.decode('utf-8'))[0]

        new_url = url.replace('.git', '')
        repo = LeanGitRepo(new_url, latest_commit)
        config = repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        if generate_benchmark_lean4.is_supported_version(v):
            print(f"Latest commit compatible for url {url}")
            return latest_commit, v

        # Search for latest commit on v4.9.1
        print(f"Searching for compatible commit for {url}")
        process = subprocess.Popen(
            ["git", "fetch", "--depth=1000000", url],  # Fetch commits
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Git fetch command failed: {stderr.decode('utf-8')}")
        process = subprocess.Popen(
            ["git", "log", "--format=%H", "FETCH_HEAD"],  # Get list of commits
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Git log command failed: {stderr.decode('utf-8')}")
        commits = stdout.decode('utf-8').strip().split('\n')
        for commit in commits:
            new_url = url.replace('.git', '')
            repo = LeanGitRepo(new_url, commit)
            config = repo.get_config("lean-toolchain")
            v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
            if generate_benchmark_lean4.is_supported_version(v):
                return commit, v

        raise Exception("No compatible commit found")

    except Exception as e:
        print(f"Error in get_compatible_commit: {str(e)}")
        return None, None

def generate_dataset(unique_urls):
    # TODO: optimize to not make novel_premises split
    # TODO: no need for licenses or metadata
    # TODO: do we need this leangitrepo stuff?
    print(f"Generating {len(unique_urls)} datasets")
    for url in unique_urls:
        if not url.endswith('.git'):
            url = url + '.git'

        sha, v = get_compatible_commit(url)
        if not sha:
            print(f"Failed to find a compatible commit for {url}")
            continue
        print(f"Found compatible commit {sha} for {url}")
        print(f"Lean version: {v}")

        url = url.replace('.git', '')
        repo = LeanGitRepo(url, sha)
        dir_name = repo.url.split("/")[-1] + "_" + sha
        dst_dir = ROOT_DIR + "/" + DATA_DIR + "/" + dir_name
        generate_benchmark_lean4.main(repo.url, sha, dst_dir)
        print(f"Finished generating benchmark at {dst_dir}")
    print("Merging datasets")
    merge_datasets()
    print("Finished merging datasets")


def get_recent_pl_checkpoint():
    response = requests.get(f"{HUGGINGFACE_API_URL}?author={USER}", timeout=10)
    models = response.json()

    if not models:
        print("No checkpoints found")
        return None

    sorted_models = sorted(models, key=lambda x: x['createdAt'], reverse=True)
    for model in sorted_models:
        if "pl" in model['modelId']:
            return model['modelId']
    
    print("No PL checkpoints found")
    return None

def download_pl_checkpoint(model_id):
    save_directory = ROOT_DIR + "/" + CHECKPOINT_DIR
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    try:
        cache_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")
        print(f"Checkpoint downloaded to: {cache_path}")
        return cache_path
    except Exception as e:
        print(f"Error downloading or saving checkpoint: {e}")
        return None

def upload_best_models_to_hf(pl_path, ct2_path, embeddings_path, next_suffix):
    api = HfApi(token=HUGGINGFACE_TOKEN)
    try:
        pl_model_name = f"pl-leancopilot-{next_suffix}"
        pl_repo_id = f"{USER}/{pl_model_name}"
        api.create_repo(repo_id=pl_repo_id, token=api.token, private=False, exist_ok=True)
        
        upload_file(
            path_or_fileobj=pl_path,
            path_in_repo="model.ckpt",
            repo_id=pl_repo_id,
            token=api.token
        )
        
        print(f"Uploaded PyTorch Lightning checkpoint to {pl_repo_id}")

        ct2_model_name = f"ct2-leancopilot-{next_suffix}"
        ct2_repo_id = f"{USER}/{ct2_model_name}"
        api.create_repo(repo_id=ct2_repo_id, token=api.token, private=False, exist_ok=True)
        upload_folder(
            folder_path=ct2_path,
            repo_id=ct2_repo_id,
            token=api.token
        )

        print(f"Uploaded CTranslate2 checkpoint to {ct2_repo_id}")

        emb_model_name = f"emb-leancopilot-{next_suffix}"
        emb_repo_id = f"{USER}/{emb_model_name}"
        api.create_repo(repo_id=emb_repo_id, token=api.token, private=False, exist_ok=True)
        upload_folder(
            folder_path=embeddings_path,
            repo_id=emb_repo_id,
            token=api.token
        )
        print(f"Uploaded embeddings to {emb_repo_id}")

        print(f"Successfully uploaded all files to separate repositories")
        return pl_repo_id, ct2_repo_id, emb_repo_id
    except Exception as e:
        print(f"Error uploading checkpoint to Hugging Face: {e}")
        return None
    
def convert_to_hf(best_model_path, hf_path) -> None:
    device = torch.device("cpu")
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "max_seq_len": 512,
        "num_retrieved": 100,
    }
    model = PremiseRetriever.load(best_model_path, device, freeze=True, config=config)
    model.encoder.save_pretrained(hf_path)
    model.tokenizer.save_pretrained(hf_path)

def get_premise_embeddings(indexed_corpus_path, embeddings_path):
    indexed_corpus = pickle.load(open(indexed_corpus_path, "rb"))

    embeddings_tensor = indexed_corpus.embeddings
    embeddings_array = embeddings_tensor.numpy()
    embeddings_array_64 = embeddings_array.astype(np.float64)

    npy_path = embeddings_path + "/" + "embeddings.npy"
    np.save(npy_path, embeddings_array_64)
    print(f"Embeddings saved to {npy_path}")

    all_premises = indexed_corpus.corpus.all_premises

    premise_dict = {
        index: {"full_name": premise.full_name, "path": premise.path, "code": premise.code}
        for index, premise in enumerate(all_premises)
    }

    dict_path = embeddings_path + "/" + "dictionary.json"
    json.dump(premise_dict, open(dict_path, "wt"), indent=4)
    print(f"Dictionary saved to {dict_path}")

def index_corpus(model_checkpoint_path, corpus_path, batch_size, embeddings_path):
    if not torch.cuda.is_available():
        print("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model = PremiseRetriever.load_hf(model_checkpoint_path, 2048, device)
    model.load_corpus(corpus_path)
    model.reindex_corpus(batch_size)

    pickle.dump(
        IndexedCorpus(model.corpus, model.corpus_embeddings.to(torch.float32).cpu()),
        open(embeddings_path, "wb"),
    )
    print(f"Indexed corpus saved to {embeddings_path}")

def convert_and_upload_models(best_model_path, new_data_path, next_suffix):
    if best_model_path:
        print(f"Best model path: {best_model_path}")

        print("Converting best model to HuggingFace format")
        hf_path = ROOT_DIR + "/" + CHECKPOINT_DIR + "/" + "hf-leancopilot-" + str(next_suffix)
        convert_to_hf(best_model_path, hf_path)
        print(f"Converted model to HuggingFace format at: {hf_path}")

        print("Converting HuggingFace model to CTranslate2 format")
        ct2_path = ROOT_DIR + "/" + CHECKPOINT_DIR + "/" + "ct2-leancopilot-" + str(next_suffix)
        convert_t5encoder_to_ct2.main(hf_path, ct2_path)
        print(f"Converted HuggingFace model to CTranslate2 format at: {ct2_path}")

        # TODO: no need to pickle and then unpickle right after
        print("Indexing the corpus to get premise embeddings")
        corpus_path = new_data_path + "/corpus.jsonl"
        batch_size = 64
        indexed_corpus_path = ROOT_DIR + "/" + CHECKPOINT_DIR + "/" + "indexed_corpus.pkl"
        index_corpus(best_model_path, corpus_path, batch_size, indexed_corpus_path)
        print("Finished indexing the corpus")

        print("Getting premise embeddings")
        embeddings_path = ROOT_DIR + "/" + CHECKPOINT_DIR + "/" + "emb-leancopilot-" + str(next_suffix)
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        get_premise_embeddings(indexed_corpus_path, embeddings_path)
        print(f"Premise embeddings saved to: {embeddings_path}")

        print("Uploading best models to Hugging Face as PL, CT2, and embeddings")
        pl_repo_id, ct2_repo_id, emb_repo_id = upload_best_models_to_hf(best_model_path, ct2_path, embeddings_path, next_suffix)
        if pl_repo_id and ct2_repo_id and emb_repo_id:
            print(f"Uploaded PyTorch Lightning model to: {pl_repo_id}")
            print(f"Uploaded CTranslate2 model to: {ct2_repo_id}")
            print(f"Uploaded embeddings to: {emb_repo_id}")
        else:
            print("Failed to upload one or more models to Hugging Face")
    else:
        print("No best model found")
    
# TODO: reduce duplication with main.py
def train(model_checkpoint_path, new_data_path, next_suffix, max_epochs=1): # TODO: chagne back to 2
    print(f"Training model with checkpoint: {model_checkpoint_path}")
    seed_everything(3407)

    if not torch.cuda.is_available():
        print("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # TODO: reduce repetition in code like this
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "max_seq_len": 512,
        "num_retrieved": 100,
    }

    model = PremiseRetriever.load(
        model_checkpoint_path, device, freeze=False, config=config
    )
    print(f"Loaded premise retriever at {model_checkpoint_path}")

    corpus_path = new_data_path + "/corpus.jsonl"
    data_path = new_data_path + "/random"
    print(f"Data path: {data_path}")
    data_module = RetrievalDataModule(
        data_path=data_path,
        corpus_path=corpus_path,
        num_negatives=3,
        num_in_file_negatives=1,
        model_name="google/byt5-small",
        batch_size=4,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=4
    )
    data_module.setup(stage='fit')

    # TODO: find a way to save lightning logs so we can view change over runs
    checkpoint_callback = ModelCheckpoint(
        dirpath=ROOT_DIR + "/" + CHECKPOINT_DIR,
        filename="_{epoch}-{Recall@10_val:.2f}",
        verbose=True,
        save_top_k=1,
        save_last=False,
        monitor="Recall@10_val",
        mode="max"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="Recall@10_val",
        patience=5,
        mode="max",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    print(f"Training dataset size after load: {len(data_module.ds_train)}")
    print(f"Validation dataset size after load: {len(data_module.ds_val)}")
    print(f"Testing dataset size after load: {len(data_module.ds_pred)}")

    trainer = pl.Trainer(
        accelerator="gpu",
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        strategy="ddp",
        devices=1, # TODO: change for GPU
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
        max_epochs=max_epochs,
        log_every_n_steps=1, # TODO: change?
        num_sanity_val_steps=0, # TODO: remove later
    )

    print("Starting progressive training...")

    trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)

    print("Finished training")

    best_model_path = checkpoint_callback.best_model_path
    print("Converting and uploading models")
    convert_and_upload_models(best_model_path, new_data_path, next_suffix)

def fetch_urls_from_api(api_url):
    try:
        response = requests.get(f"{api_url}/get_urls/")
        response.raise_for_status()
        return response.json()["urls"]
    except requests.RequestException as e:
        print(f"Error fetching URLs from API: {e}")
        return []

def main():
    # TODO: close instance on done
    # TODO: should we close instance on failure?
    # TODO: put a try-catch around generate benchmark, and anything else that could fail, same with main.py
    # TODO: should we just not cache the datasets so we can save space?
    if ray.is_initialized():
        ray.shutdown()
    
    if not os.path.exists(ROOT_DIR + "/" + DATA_DIR):
        os.makedirs(ROOT_DIR + "/" + DATA_DIR)

    print("GitHub Token:", os.environ.get('GITHUB_ACCESS_TOKEN')) # TODO: remove
    api_url = "https://leancopilotapi.onrender.com" # TODO: update later
    unique_urls = set(fetch_urls_from_api(api_url))
    print(f"Unique URLs: {unique_urls}")

    return # TODO: remove

    generate_dataset(unique_urls)

    model_id = get_recent_pl_checkpoint()
    if model_id:
        print(f"Latest PL checkpoint found: {model_id}")
        model_checkpoint_path = download_pl_checkpoint(model_id)
        if model_checkpoint_path:
            print(f"Checkpoint path: {model_checkpoint_path}")
            merged_data_path = ROOT_DIR + "/" + DATA_DIR + "/" + "merged"
            next_suffix = int(model_id.split("-")[-1]) + 1
            train(model_checkpoint_path, merged_data_path, next_suffix)
        else:
            print("Error downloading or saving checkpoint")
    else:
        print("No checkpoints found")

if __name__ == "__main__":
    main()
