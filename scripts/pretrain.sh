#!/usr/bin/env bash
#
# Author: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
# Copyright (c) 2020. Johns Hopkins University - All rights reserved.
#

CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 15\
                --batch_size 1\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 2
                --dataset sceneflow\
                --dataset_directory PATH_TO_SCENEFLOW