import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)
        self.log('Recall@10_val', torch.rand(1).item() * 100)  # Dummy metric
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def create_dummy_data():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    return TensorDataset(x, y)

def main():
    model = DummyModel()

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='repo_{epoch:02d}-{Recall@10_val:.2f}',
        verbose=True,
        save_top_k=-1,
        every_n_epochs=1,
        monitor='Recall@10_val',
        mode='max'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        enable_checkpointing=True
    )

    # Dataset 1 (simulating first repository)
    dataset1 = create_dummy_data()
    train_loader1 = DataLoader(dataset1, batch_size=32)
    val_loader1 = DataLoader(dataset1, batch_size=32)

    print("Training on dataset 1")
    trainer.fit(model, train_loader1, val_loader1)

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        enable_checkpointing=True
    )

    # Dataset 2 (simulating second repository)
    dataset2 = create_dummy_data()
    train_loader2 = DataLoader(dataset2, batch_size=32)
    val_loader2 = DataLoader(dataset2, batch_size=32)

    print("Training on dataset 2")
    trainer.fit(model, train_loader2, val_loader2)

    # Print out all saved checkpoints
    print("\nSaved checkpoints:")
    for filename in os.listdir('checkpoints'):
        print(filename)
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")

if __name__ == "__main__":
    main()