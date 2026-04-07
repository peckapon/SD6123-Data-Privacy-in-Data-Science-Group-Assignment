#!/bin/bash
#SBATCH --job-name=centralized_gpu
#SBATCH --output=centralized_output.log
#SBATCH --error=centralized_error.log
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gpus=1

echo "Running on $(hostname)"
nvidia-smi

source /{PATH}/venvs/fednova310/bin/activate

PYTHONPATH=$(pwd) python -m fednova.centralized