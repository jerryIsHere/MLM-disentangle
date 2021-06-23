#!/bin/bash
#SBATCH --qos=debug
#SBATCH --partition=debugq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs1/home/ckchan666/job/check_custom_xtreme/task.log      # Standard output
#SBATCH --error=/gpfs1/home/ckchan666/job/check_custom_xtreme/task.err       # Standard error log


module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/test/check_custom_xtreme.py
