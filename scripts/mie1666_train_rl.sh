#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=15:59:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-khalile2
#SBATCH --mail-user=changy.liu@mail.utoronto.ca
#SBATCH --mail-type=ALL


echo "Running on Graham cluster"

# module load NiaEnv/2019b
# module use /scinet/niagara/software/commercial/modules

module load python/3.8
# export PYTHONPATH=$PYTHONPATH:/scinet/niagara/software/commercial/gurobi951/linux64/lib/python3.8

# export MODULEPATH=$HOME/modulefiles:$MODULEPATH
# module load mycplex/12.8.0

# module load gurobi/9.5.1

source /home/liucha90/chang_pytorch/bin/activate

python3.8 runMIPall.py --dataset_name "${dataset_name:="1PDPTW_generated_d21_i1000_tmin300_tmax500_sd2022_test"}"
