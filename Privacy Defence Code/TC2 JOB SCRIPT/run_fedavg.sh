#!/bin/bash
#SBATCH --job-name=fedavg_gpu
#SBATCH --output=fedavg_output.log
#SBATCH --error=fedavg_error.log
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gpus=1

echo "Running on $(hostname)"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# auto-pick GPU with most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
| nl -v 0 | sort -nrk2 | head -n1 | awk '{print $1}')

source /{PATH}/venvs/fednova310/bin/activate
PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fedavg optimizer=fedavg_opt