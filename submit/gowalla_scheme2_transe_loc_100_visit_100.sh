#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J gowalla_scheme2_transe_loc_100_visit_100
#SBATCH -o gowalla_scheme2_transe_loc_100_visit_100.%J.out
#SBATCH -e gowalla_scheme2_transe_loc_100_visit_100.%J.err
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
/home/hanp/anaconda3/envs/rx_torch/bin/python train.py --trans_loc_file KGE/gowalla_scheme2_transe_loc_temporal_100.pkl --trans_interact_file KGE/gowalla_scheme2_transe_user-loc_100.pkl
