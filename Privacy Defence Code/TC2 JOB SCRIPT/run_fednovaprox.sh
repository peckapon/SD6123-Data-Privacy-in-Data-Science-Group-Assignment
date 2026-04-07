#!/bin/bash
#SBATCH --job-name=fednovaprox_gpu
#SBATCH --output=fednovaprox_output.log
#SBATCH --error=fednovaprox_error.log
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gpus=1

echo "Running on $(hostname)"
nvidia-smi

source /{PATH}/venvs/fednova310/bin/activate

PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fednova optimizer=fednova_prox_opt