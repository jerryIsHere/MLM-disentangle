#!/bin/bash
#SBATCH --partition=gpu_7d1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=0-50:00:00
#SBATCH --output=/gpfs1/home/ckchan666/xnli_disentangle_experiment/job/xnli/task.log
#SBATCH --error=/gpfs1/home/ckchan666/xnli_disentangle_experiment/job/xnli/task.err

module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/experiment/xnli/xnli_experiment.py