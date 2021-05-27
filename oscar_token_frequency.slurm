#!/bin/bash
#SBATCH --partition=cpu_14d1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=64
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.log
#SBATCH --error=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.err
# 65536

module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/oscar_token_frequency.py
