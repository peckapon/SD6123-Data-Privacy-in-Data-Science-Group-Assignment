#!/bin/bash
#SBATCH --job-name=fedprox_gpu
#SBATCH --output=fedprox_output.log
#SBATCH --error=fedprox_error.log
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gpus=1

echo "Running on $(hostname)"
nvidia-smi

source /{PATH}/venvs/fednova310/bin/activate

PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fedavg optimizer=fedprox_opt