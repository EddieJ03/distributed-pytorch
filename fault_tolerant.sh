#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 FaultTolerantDistributedTraining.py
