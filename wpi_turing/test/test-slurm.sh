#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=5g
#SBATCH -J "Test S3 and Cuda"
#SBATCH -p academic
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100

# Load CUDA
module load cuda12.2

echo "running source activate"
# Activate conda env, source activate works but not conda activate
source activate cnn-pe

# Run the job
echo "starting script now"
cd ..
python test/test_slurm.py
echo "I am done"
