#!/bin/bash
#SBATCH --partition=gpu_7d1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=0-05:00:00
#SBATCH --output=/gpfs1/home/ckchan666/mlm_disentangle_experinment/job/mlm/task.log
#SBATCH --error=/gpfs1/home/ckchan666/mlm_disentangle_experinment/job/mlm/task.err

module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/experinment/mlm/mlm_experinment.py 04:00:00 --config_json /gpfs1/home/ckchan666/MLM-disentangle/experinment/mlm/debug.json
