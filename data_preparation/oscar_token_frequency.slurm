#!/bin/bash
#SBATCH --partition=cpu_14d1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.log
#SBATCH --error=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.err
# 65536

module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/oscar_token_frequency.py
