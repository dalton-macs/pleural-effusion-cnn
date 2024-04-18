#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=12g
#SBATCH -J "resnet checkpoint"
#SBATCH -p academic
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100

# Set PYTHONPATH to include the root project directory
export PYTHONPATH="/home/dgmacres/projects/pleural-effusion-cnn:$PYTHONPATH"

# Load CUDA
module load cuda12.2

# Activate conda env, source activate works but not conda activate
source activate cnn-pe

# Run the job
echo "starting script now"

cd ../..
pwd

# python src/train.py resnet
python src/train.py resnet --checkpoint-path "model/ResNet18ShowkatCustom/checkpoints/epoch_27_batch_233.pth"

echo "I am done"

source deactivate
