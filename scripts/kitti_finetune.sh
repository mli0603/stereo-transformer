#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 400\
                --batch_size 1\
                --checkpoint kitti_ft\
                --num_workers 2\
                --dataset kitti\
                --dataset_directory PATH_TO_KITTI\
                --ft\
                --resume sceneflow_pretrained_model.pth.tar