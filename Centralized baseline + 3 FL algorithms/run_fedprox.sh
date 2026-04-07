#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --job-name=FL_fedprox
#SBATCH --output=/{PATH}/%j.out
#SBATCH --error=/{PATH}/%j.err

module load anaconda/25.5.1
module load cuda/12.8.0

eval "$(conda shell.bash hook)"
conda activate FLEnv

cd /{PATH}/fednova

export PYTHONPATH=/{PATH}/fednova:$PYTHONPATH

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

python fednova/main.py num_rounds=100 strategy=fedavg optimizer=fedprox_opt
