import pytorch_lightning as pl
from loguru import logger
import torch
import torch.distributed as dist
import pickle

class FisherComputationModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fisher_info = {}
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        self.model.zero_grad()
        loss = self.model(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )

        self.manual_backward(loss)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.fisher_info:
                    self.fisher_info[name] = param.grad.detach().clone() ** 2
                else:
                    self.fisher_info[name] += param.grad.detach().clone() ** 2

        return loss

    def on_train_epoch_end(self):
        logger.info("Synchronizing and normalizing Fisher Information")
        
        # Synchronize Fisher Information across GPUs
        # Each GPU now has the sum of the Fisher Information from all GPUs for each parameter
        for name in self.fisher_info:
            dist.all_reduce(self.fisher_info[name], op=dist.ReduceOp.SUM)

        logger.info("Normalizing Fisher Information")
        logger.info(f"length of dataset: {len(self.trainer.train_dataloader.dataset)}")
        world_size = dist.get_world_size()
        logger.info(f"world_size: {world_size}")
        dataset_size = len(self.trainer.train_dataloader.dataset)
        logger.info(f"dataset_size: {dataset_size}")
        total_samples = dataset_size * world_size
        logger.info(f"total_samples: {total_samples}")
        for name in self.fisher_info:
            self.fisher_info[name] /= total_samples

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0)  # We don't actually want to update the model

    def save_fisher_info(self, fisher_file_path):
        if self.trainer.is_global_zero:
            logger.info(f"Saving Fisher Information Matrix to {fisher_file_path}")
            with open(fisher_file_path, "wb") as f:
                    pickle.dump(self.fisher_info, f)