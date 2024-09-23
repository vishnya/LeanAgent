import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ddp_test import MNISTModel, MNISTDataModule

def main():
    is_main_process = int(os.environ.get('LOCAL_RANK', '0')) == 0
    print("at the beginning separate")
    print("is_main_process separate: ", is_main_process)
    
    # Initialize the model and data module
    model = MNISTModel()
    data_module = MNISTDataModule()

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=2,  # Use 2 GPUs
        strategy='ddp',  # Use Distributed Data Parallel
        num_nodes=1,  # Use a single machine
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

if __name__ == "__main__":
    print("here separate")
    main()