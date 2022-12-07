#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --output=%N-%j.out
#SBATCH --partition=compute
#SBATCH --account=def-beck

echo "Running on Graham cluster"

# module load NiaEnv/2019b
# module use /scinet/niagara/software/commercial/modules

module load python/3.8
# export PYTHONPATH=$PYTHONPATH:/scinet/niagara/software/commercial/gurobi951/linux64/lib/python3.8

# export MODULEPATH=$HOME/modulefiles:$MODULEPATH
# module load mycplex/12.8.0

# module load gurobi/9.5.1

source /home/b/beck/minori/my_env2/bin/activate

emb_dim=${1}
emb_iter_T=${2}
num_episodes= ${3}
mem_capacity= ${4}
n_step_ql= ${5}
batch_size= ${6}
gamma=${7}
lr= ${8}
lr_decay_rate= ${9}
min_epsilon= ${10}
epsilon_decay_rate= ${11}
beta= ${12}
dataset_name= ${13}
seed= ${14}

python3 train.py --emb_dim "${emb-dim}" \
                --emb_iter_T "${emb_iter-T}" \
                --num_episodes "${num-episodes}" \
                --mem_capacity "${mem-capacity}" \
                --n_step_ql "${n-step-ql}" \
                --batch_size "${batch-size}" \
                --gamma "${gamma}" \
                --lr "${lr}" \
                --lr_decay_rate "${lr-decay-rate}" \
                --min_epsilon "${min-epsilon}" \
                --epsilon_decay_rate "${epsilon-decay-rate}" \
                --beta "${beta}" \
                --dataset_name "${dataset-name}"\
                --seed "${seed}"

