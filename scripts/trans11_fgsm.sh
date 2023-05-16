#!/usr/bin/env bash
#SBATCH --time=4:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=230G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/fgsm_new/1again_trans_9_freq_%A.out


CUDA_VISIBLE_DEVICES=0 python main_fgsm.py  --epochs 15 --batch_size 1 --checkpoint final_trans_9_fgsm_freq --pre_train --num_workers 64 --dataset sceneflow --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW --resume /work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/trans_9/experiment_3/epoch_14_model.pth.tar --eval --fgsm --kernel_size 9 --epsilon $1

