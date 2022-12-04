#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --output=%N-%j.out
#SBATCH --partition=compute
#SBATCH --account=def-beck

echo "Running on Niagara cluster"

module load NiaEnv/2019b
module use /scinet/niagara/software/commercial/modules

module load python/3.8
export PYTHONPATH=$PYTHONPATH:/scinet/niagara/software/commercial/gurobi951/linux64/lib/python3.8

export MODULEPATH=$HOME/modulefiles:$MODULEPATH
module load mycplex/12.8.0

module load gurobi/9.5.1

source /home/b/beck/minori/my_env2/bin/activate

if [ $# -ne 19 ]; then
    echo "Number of arguments passed was $#." 1>&2
    echo "To execute this script, you need to pass 15 arguments." 1>&2
    exit 1
fi
method=${1}
dataset_path=${2}
train_dataset_path=${3}
k=${4}
time_limit=${5}
max_workers=${6}
batch_size=${7}
max_iter=${8}
thread=${9}
use_job_id=${10}
output_type=${11}
model_type=${12}
teacher_forcing_ratio=${13}
train_dummy_p=${14}
train_dummy_d=${15}
test_dummy_p=${16}
test_dummy_d=${17}
resume=${18}
parallel=${19}

if [ $(($thread * $max_workers)) -gt 80 ]; then
    echo "Required resources excessed maximum limit (80 cores)."
    echo "Make sure max_workers times thread does not exceed 80."
    exit 1
fi

cd ../

mkdir -p ./tmp

if [ $(($resume)) = 1 ]; then
    resume='--resume'
else
    resume=''
fi

if [ $(($use_job_id)) = 1 ]; then
    use_job_id='--use-job-id'
else
    use_job_id=''
fi

if [ $(($parallel)) = 1 ]; then
    parallel='--parallel'
else
    parallel=''
fi

python3 solve_all_json.py --method "${method}" --train-dataset-path "${train_dataset_path:=Dataset_NN_d_s1000_v3}" \
                        --dataset-path "${dataset_path:=Dataset_NN_d_s1000_v3_test}" --k "${k:=10}" \
                        --max-workers "${max_workers:=10}" --batch-size "${batch_size:=64}" \
                        --max-iter "${max_iter:=1000}" --output-type "${output_type:=idx}" --thread "${thread:=8}"\
                        --train-dummy-p "${train_dummy_p:=300}" --train-dummy-d "${train_dummy_d:=5000}" \
                        --test-dummy-p "${test_dummy_p:=300}" --test-dummy-d "${test_dummy_d:=5000}" \
                        "${use_job_id}" "${resume}" "${parallel}" --model-type "${model_type:=rnn}" \
                        --teacher-forcing-ratio "${teacher_forcing_ratio:=0.5}" --time-limit "${time_limit:=0}"
