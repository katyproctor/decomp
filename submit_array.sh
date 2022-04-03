#!/bin/bash 
#SBATCH --job-name=group_arr
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --array=0-201

#SBATCH --output="./logs/process_%a.out"
#SBATCH --error="./logs/process_%a.err"

readarray -t FILES < ./groups_L0050.txt

module load gcc/9.2.0 openmpi/4.0.2 hdf5/1.12.0 python/3.7.4
source ~/venv/bin/activate

python main_array.py -gl ${FILES[@]:$SLURM_ARRAY_TASK_ID*999:999}
