#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/fgsm/trans_7_%A.out

reload
sttr

CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 15\
                --batch_size 1\
                --checkpoint trans7\
                --pre_train\
                --num_workers 64\
                --dataset sceneflow\
                --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW\
                --kernel_size 7\
                --resume /work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/trans7/experiment_3/epoch_14_model.pth.tar\
                --eval\
                --fgsm\
                --epsilon $1

