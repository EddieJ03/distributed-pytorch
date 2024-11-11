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



def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)

    # nccl is NVIDIA collective communications lib, popular backend for communication btwn nvidia GPUs
    init_process_group(backend="nccl", rank=rank, world_size=world_size)



def multi_gpu(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32, distributed=True)
    
    trainer = MyTrainer(model, train_data, optimizer, True, False, rank, save_every, "")
    trainer.train(total_epochs)
    
    destroy_process_group()



# start training job

if __name__ == '__main__':
    total_epochs = 50
    save_every = 2
    start_time = time.time()
    world_size = torch.cuda.device_count()
    print('world size is : ', world_size)
    mp.spawn(multi_gpu, args=(world_size, total_epochs, save_every,), nprocs=world_size) 
    end_time = time.time()
    print(f"Multi GPU Execution time: {end_time - start_time:.4f} seconds")    



