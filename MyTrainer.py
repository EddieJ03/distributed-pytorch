#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from Encoder import Classifier

class MyTrainer:
    def __init__(
        self,
        model: Classifier,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        distributed: bool,
        fault_tolerant: bool,
        gpu_id: int,
        save_every: int,
        snapshot_path: str
    ) -> None:
        self.gpu_id = gpu_id

        if fault_tolerant:
            print('Torch Run Fault Tolerant')
            self.gpu_id = int(os.environ["LOCAL_RANK"])

        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.distributed = distributed
        self.fault_tolerant = fault_tolerant
        self.epochs_run = 0

        if self.fault_tolerant and os.path.exists(snapshot_path):
            print("LOADING SNAPSHOT")
            self._load_snapshot(snapshot_path)

        if distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            
        self.save_every = save_every
    
    # only gets called for fault tolerant runs
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output, _ = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        
        train_loss = 0
        
        self.model.train()
        
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            train_loss += self._run_batch(source, targets)
        
        return train_loss

    def _save_checkpoint(self, epoch):
        ckp = None

        if self.distributed:
            ckp = self.model.module.state_dict()
        else:
            ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            epoch_loss = self._run_epoch(epoch)
            
            if (not self.distributed or self.gpu_id == 0) and epoch % self.save_every == 0:
                print('Train Loss: ', epoch_loss)
                
                if self.fault_tolerant:
                    self._save_snapshot(epoch)
                else:
                    self._save_checkpoint(epoch)


