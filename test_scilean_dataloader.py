import os
import json
from pathlib import Path
from tqdm import tqdm
from retrieval.datamodule import RetrievalDataModule
from common import Premise

RAID_DIR = os.environ.get('RAID_DIR', '/raid/adarsh')
DATA_DIR = "datasets_test"

def test_scilean_dataloader():
    # Find the SciLean dataset
    dataset_path = Path(RAID_DIR) / DATA_DIR
    scilean_dir = next(d for d in dataset_path.iterdir() if 'SciLean' in d.name)
    
    print(f"Testing SciLean dataset: {scilean_dir}")

    # Set up the data module
    data_module = RetrievalDataModule(
        data_path=str(scilean_dir / "random"),
        corpus_path=str(scilean_dir / "corpus.jsonl"),
        num_negatives=3,
        num_in_file_negatives=1,
        model_name="google/byt5-small",
        batch_size=4,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=4
    )

    # Setup the data module
    data_module.setup(stage='fit')

    # Test the train dataloader
    print("Testing train dataloader...")
    for batch in tqdm(data_module.train_dataloader()):
        pass

    print("Testing validation dataloader...")
    for batch in tqdm(data_module.val_dataloader()):
        pass

    print("Testing test dataloader...")
    for batch in tqdm(data_module.test_dataloader()):
        pass

    print("All dataloaders processed successfully!")

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
                    kind=p['kind']
                )
                try:
                    serialized = premise.serialize()
                except Exception as e:
                    print(f"Error serializing premise: {premise.full_name}")
                    print(f"Error message: {str(e)}")
                    print(f"Premise code: {premise.code}")
                    raise

    print("All premises serialized successfully!")

if __name__ == "__main__":
    test_scilean_dataloader()