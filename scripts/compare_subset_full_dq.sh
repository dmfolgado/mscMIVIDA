#!/bin/bash
#SBATCH --job-name=summarization_data_quality
#SBATCH --output=summarization_data_quality.log
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=duarte.folgado@fraunhofer.pt
#SBATCH --mail-type=ALL

# GPU diagnostic
nvidia-smi

# Set CPU thread usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Execute data quality evaluation
python src/mscmivida/evaluation/data_quality_deepcore.py

echo "Subset data quality evaluation terminated successfully."
