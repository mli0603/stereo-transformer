#!/usr/bin/env bash
#SBATCH --time=23:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/training_new/trans_9_%A.out          

CUDA_VISIBLE_DEVICES=0 python main.py --epochs 15 --batch_size 1 --checkpoint trans_9 --pre_train --num_workers 64 --dataset sceneflow --kernel_size 9 --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW --resume /work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/trans_9/experiment_2/epoch_10_model.pth.tar

