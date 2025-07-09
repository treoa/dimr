import torch

import pytorch_lightning as pl

class Dataset(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        self.train_dataset = torch.utils.data.Dataset(
            self.data_dir,
            self.batch_size,
            self.num_workers
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)