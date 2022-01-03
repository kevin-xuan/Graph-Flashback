#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J submit_transh_hidden_dim_50
#SBATCH -o submit_transh_hidden_dim_50.%J.out
#SBATCH -e submit_transh_hidden_dim_50.%J.err
#SBATCH --mail-user=peng.han@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=48:30:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=[gpu]

#run the application:
module load cudnn/7.5.0-cuda10.1.105
module load nccl/2.4.8-cuda10.1
cd /home/hanp/hanp/research/UESTC/xuan_rao/new_Flashback_code 
/home/hanp/anaconda3/envs/rx_torch/bin/python train.py --trans_loc_file KGE/scheme1_transh_loc_temporal_20.pkl --hidden-dim 50 --weight_decay 0.00001
