import os
import shutil
import unittest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class TestPyTorchLightningLogging(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath('test_lightning_logs')
        print(f"Test directory: {self.test_dir}")
        if os.path.exists(self.test_dir):
            print(f"Removing existing test directory: {self.test_dir}")
            shutil.rmtree(self.test_dir)
        print(f"Creating test directory: {self.test_dir}")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            print(f"Cleaning up test directory: {self.test_dir}")
            shutil.rmtree(self.test_dir)

    def test_custom_logging_directory(self):
        print("Starting test_custom_logging_directory")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Parent directory contents: {os.listdir(os.path.dirname(self.test_dir))}")
        print(f"Test directory permissions: {oct(os.stat(self.test_dir).st_mode)[-3:]}")
        
        # Create a simple dataset
        x = torch.linspace(0, 10, 100).reshape(-1, 1)
        y = 2 * x + torch.randn_like(x)
        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=10)

        # Initialize model
        model = SimpleModel()

        # Setup trainer with custom log directory
        trainer = pl.Trainer(
            max_epochs=1,
            default_root_dir=self.test_dir,
            enable_checkpointing=False,
            logger=True,
            log_every_n_steps=1,
            accelerator="auto",
            devices="auto",
            strategy="ddp",  # Use DistributedDataParallel strategy
        )

        print(f"Trainer initialized with default_root_dir: {self.test_dir}")

        # Train the model
        trainer.fit(model, train_loader)

        print(f"Training completed. Checking log directory: {self.test_dir}")

        # Check if log directory was created
        self.assertTrue(os.path.exists(self.test_dir), f"Test directory {self.test_dir} does not exist")
        
        version_dirs = [d for d in os.listdir(self.test_dir) if d.startswith('version_')]
        print(f"Version directories found: {version_dirs}")
        self.assertTrue(len(version_dirs) > 0, "No version directory created")

        # Check if logs were created
        log_dir = os.path.join(self.test_dir, version_dirs[0])
        log_files = os.listdir(log_dir)
        print(f"Files in log directory: {log_files}")
        self.assertTrue(any(file.endswith('.log') for file in log_files), "No log file created")
        self.assertTrue(any(file.startswith('events.out.tfevents') for file in log_files), "No TensorBoard event file created")

        print("Test completed successfully")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    unittest.main(verbosity=2)