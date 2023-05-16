#!/usr/bin/env bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/training_new/trans_7_3_2_%A.out
          

CUDA_VISIBLE_DEVICES=0 python main.py  --epochs 20 --batch_size 1 --checkpoint trans7_3_2 --pre_train --num_workers 64 --dataset sceneflow --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW --kernel_size 7 -pk --epsilon 0.0 --resume /work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/trans7_3/experiment_2/epoch_14_model.pth.tar

