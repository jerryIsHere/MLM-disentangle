#!/bin/bash
#SBATCH --partition=gpu_7d1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.log
#SBATCH --error=/gpfs1/home/ckchan666/job/oscar_token_frequency/task.err

module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/experinment/mlm_disentangle/mlm_disentanglement_experinment.py 01:00:00
