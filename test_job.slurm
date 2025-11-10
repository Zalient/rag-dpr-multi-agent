#!/bin/bash

# --- Job Configuration ---
#SBATCH -J test_1b_job                    # Job name
#SBATCH -p l4_nodes                       # Partition (queue) to use L4 nodes (as per supervisor)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH -t 00:15:00                       # Time limit: 15 minutes (plenty for this test)
#SBATCH --mem=32G                         # Memory request

# --- Output & Email (as per supervisor) ---
#SBATCH -o ~/slurm_logs/test_1b_job-%j.out  # Save output to scratch
#SBATCH -e ~/slurm_logs/test_1b_job-%j.err  # Save errors to scratch
#SBATCH --mail-type=END,FAIL              # Email on job end or failure
#SBATCH --mail-user=zc3g23@soton.ac.uk    # Your email address

# --- Job Steps ---
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Username: $USER"
mkdir -p /local/scratch/users/$USER
echo "Scratch directory: /local/scratch/users/$USER"

# --- Activate Conda Environment ---
# (Assumes miniconda3 is in your home directory)
echo "Activating Conda..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag-dpr-multi-agent
echo "Conda environment 'rag-dpr-multi-agent' activated."

# --- Run the Python Test Script ---
echo "Running Python test script..."
python test_1b_model.py
echo "Python script finished."

echo "Job finished at $(date)"