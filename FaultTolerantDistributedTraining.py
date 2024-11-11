#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset
from MyTrainer import MyTrainer
from MyTrainDataset import MyTrainDataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
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



def ddp_setup():
    # nccl is NVIDIA collective communications lib, popular backend for communication btwn nvidia GPUs
    init_process_group(backend="nccl")



def multi_gpu(total_epochs, save_every):
    ddp_setup()
    
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32, distributed=True)
    
    trainer = MyTrainer(model, train_data, optimizer, True, True, -1, save_every, "snapshot.pt")
    trainer.train(total_epochs)
    
    destroy_process_group()


# start training job

if __name__ == '__main__':
    total_epochs = 50
    save_every = 2
    start_time = time.time()
    multi_gpu(total_epochs, save_every) 
    end_time = time.time()
    print(f"Multi GPU Execution time: {end_time - start_time:.4f} seconds")    



