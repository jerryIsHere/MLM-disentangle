#!/bin/bash
#SBATCH --qoa=debug
#SBATCH --partition=debugq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.log
#SBATCH --error=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.err
# 65536

module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/oscar_token_frequency.py --example 256
