#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:2
#SBATCH --ntasks-per-node=32
#SBATCH --time=5:59:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-khalile2
#SBATCH --mail-user=changy.liu@mail.utoronto.ca
#SBATCH --mail-type=ALL

echo "Running on Graham cluster"

module load python/3.8

source /home/liucha90/chang_pytorch/bin/activate

# emb_dim=${1}
# emb_iter_T=${2}
# num_episodes= ${3}
# mem_capacity= ${4}
# n_step_ql= ${5}
# batch_size= ${6}
# gamma=${7}
# lr= ${8}
# lr_decay_rate= ${9}
# min_epsilon= ${10}
# epsilon_decay_rate= ${11}
# beta= ${12}
# dataset_name= ${13}
# seed= ${14}

# python3 train.py --emb_dim "${emb-dim}" \
#                 --emb_iter_T "${emb_iter-T}" \
#                 --num_episodes "${num-episodes}" \
#                 --mem_capacity "${mem-capacity}" \
#                 --n_step_ql "${n-step-ql}" \
#                 --batch_size "${batch-size}" \
#                 --gamma "${gamma}" \
#                 --lr "${lr}" \
#                 --lr_decay_rate "${lr-decay-rate}" \
#                 --min_epsilon "${min-epsilon}" \
#                 --epsilon_decay_rate "${epsilon-decay-rate}" \
#                 --beta "${beta}" \
#                 --dataset_name "${dataset-name}"\
#                 --seed "${seed}"


python3.8 train.py --num_episodes 30001 --dataset_name "1PDPTW_generated_d21_i100000_tmin300_tmax500_sd2022" --seed 1