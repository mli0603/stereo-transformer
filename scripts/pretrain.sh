#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 15\
                --batch_size 1\
                --checkpoint vanilla\
                --pre_train\
                --num_workers 64\
                --dataset sceneflow\
                --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW\
                --kernel_size 3

