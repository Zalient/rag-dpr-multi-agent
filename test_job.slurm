#!/bin/bash

#SBATCH -J test_1b_job          # Job name
#SBATCH -p ecsstudents_l4       # Partition (queue) to use L4 nodes
#SBATCH --nodes=1               # Number of nodes to request
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH -t 00:15:00             # Walltime (15 minutes)
#SBATCH --mem=32G               # Memory request

#SBATCH -o /scratch/zc3g23/slurm_logs/test_1b_job-%j.out  # Save output to scratch
#SBATCH -e /scratch/zc3g23/slurm_logs/test_1b_job-%j.err  # Save errors to scratch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zorclarke@gmail.com

# Job steps
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Username: $USER"

# Activate Conda environment
echo "Loading Conda..."
module load conda/python3
echo "Activating Conda..."
eval "$(conda shell.bash hook)"
conda activate rag-dpr-multi-agent
echo "Conda environment 'rag-dpr-multi-agent' activated."

# Run the Python test script
echo "Running Python test script..."
python test_1b_model.py
echo "Python script finished."

echo "Job finished at $(date)"