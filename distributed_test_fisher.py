import os
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
from datetime import timedelta
import time
from loguru import logger

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        logger.info("SimpleModel init")
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        logger.info("SimpleModel training_step")
        x, y = batch
        y_hat = self.layer(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)

def main(args):
    # Set up data
    train_data = TensorDataset(torch.randn(100, 1), torch.randn(100, 1))
    train_loader = DataLoader(train_data, batch_size=10)

    # Set up model
    model = SimpleModel()

    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_TIMEOUT'] = str(args.timeout * 1000)

    fisher_trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,  # Use all 4 GPUs
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=1,  # We only need one pass through the data
    )

    logger.info("before fisher fit")
    fisher_trainer.strategy.barrier()

    fisher_trainer.fit(model, train_loader)

    logger.info("after fisher fit")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    args = parser.parse_args()

    main(args)
