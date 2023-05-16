#!/usr/bin/env bash
#SBATCH --time=23:59:59
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/training_new/trans_11_again_%A.out          

CUDA_VISIBLE_DEVICES=0 python main.py --epochs 15 --batch_size 1 --checkpoint trans11_again --pre_train --num_workers 64 --dataset sceneflow --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW --resume /work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/trans11_again/experiment_2/epoch_14_model.pth.tar

