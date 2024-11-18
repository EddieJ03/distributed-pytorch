#!/usr/bin/env python
# coding: utf-8

import pickle
import torch
from torch.utils.data import Dataset
from MyTrainer import MyTrainer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

from Encoder import Classifier
from PresidentsDataset import SpeechesClassificationDataset
from Tokenizer import SimpleTokenizer
from Training import load_texts, collate_batch


def load_train_objs():
    tokenizer = SimpleTokenizer() # create a tokenizer from the data

    tokenizer_filename = './tokenizer.pkl'

    if os.path.exists(tokenizer_filename):
        with open(tokenizer_filename, 'rb') as file:
            tokenizer = pickle.load(file)

        print('loaded tokenizer from file')
    else:
        for text in load_texts('./'):
            tokenizer.update_vocab(text.split('\t', 1)[1])

        with open(tokenizer_filename, 'wb') as file:
            pickle.dump(tokenizer, file)

        print('created tokenizer and stored')
    
    train_set = SpeechesClassificationDataset(tokenizer, "./train.tsv")  # load your dataset
    
    model = Classifier(tokenizer.vocab_size)  # load your model
    
    optimizer = torch.optim.Adam(
        model.parameters(),  
        lr=0.0001,            
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, distributed: bool):
    return DataLoader(
        dataset,
        collate_fn=collate_batch,
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
    world_size = torch.cuda.device_count()
    print('world size is : ', world_size)
    mp.spawn(multi_gpu, args=(world_size, total_epochs, save_every,), nprocs=world_size) 



