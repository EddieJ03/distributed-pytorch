#!/usr/bin/env python
# coding: utf-8

import torch
from MyTrainer import MyTrainer
from MyTrainDataset import MyTrainDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import time

def load_train_objs():
    train_set = MyTrainDataset(4096)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, distributed: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=not distributed,
        sampler=None if not distributed else DistributedSampler(dataset),
    )

# cuda device
device = 1

# batch size
batch_size = 32

# epochs & save_every
total_epochs = 50
save_every = 2

dataset, model, optimizer = load_train_objs()
train_data = prepare_dataloader(dataset, batch_size, False)

# pass in false for NO distributed training
# trainer = Trainer(model, train_data, optimizer, False, device, save_every)

start_time = time.time()
trainer = MyTrainer(model, train_data, optimizer, False, False, device, save_every, "")
trainer.train(total_epochs)
end_time = time.time()
print(f"Single GPU Execution time: {end_time - start_time:.4f} seconds")


