#!/usr/bin/env bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/new_neurips/pgd_vanilla_%A.out

#reload
#sttr
#module unload pytorch-py37-cuda11.2-gcc8/1.9.1
#pip install typing-extensions

CUDA_VISIBLE_DEVICES=0
python main_fgsm.py  --epochs 15\
                --batch_size 1\
                --checkpoint icml_pgd\
                --pre_train\
                --num_workers 64\
                --dataset sceneflow\
                --dataset_directory /work/ws-tmp/sa058646-segment2/stereo-transformer/data/SCENE_FLOW\
                --kernel_size 3\
                --resume /work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/vanilla/experiment_3/epoch_14_model.pth.tar\
                --eval\
                --fgsm\
                --epsilon $1\
                -it $2\
                -at fgsm\
                --alpha $1

