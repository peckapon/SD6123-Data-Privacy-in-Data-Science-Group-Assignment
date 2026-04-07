#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --nodelist=TC2N07
#SBATCH --job-name=FL_centralized
#SBATCH --output=/{PATH}/%j.out
#SBATCH --error=/{PATH}/%j.err

module load anaconda/25.5.1
module load cuda/12.8.0

eval "$(conda shell.bash hook)"
conda activate FLEnv

cd /{PATH}/fednova

export PYTHONPATH=/{PATH}/fednova:$PYTHONPATH

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

python fednova/centralized.py \
    --num_rounds 100 \
    --lr 0.05 \
    --momentum 0.0 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --seed 1 \
    --datapath fednova/data/ \
    --checkpoint_path fednova/checkpoints/ \
    --output_path fednova/results/
