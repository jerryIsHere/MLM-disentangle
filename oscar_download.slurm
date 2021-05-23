#!/bin/bash
#SBATCH --partition=cpu_14d1n
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 32 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH TASK, i.e. 1x32x4=128 cores
#SBATCH --mem=8GB                # 440GB mem on EACH NODE
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --output=/gpfs1/home/ckchan666/job/oscar_download/task.log      # Standard output
#SBATCH --error=/gpfs1/home/ckchan666/job/oscar_download/task.err       # Standard error log


module load python37
source /gpfs1/home/ckchan666/mnlpEnv/bin/activate
python /gpfs1/home/ckchan666/MLM-disentangle/oscar_download.py
