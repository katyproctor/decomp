#!/bin/bash 
#SBATCH --job-name=100Mpc
#SBATCH --time=09:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --array=0-1

#SBATCH --output="./logs/100Mpc_%a.out"
#SBATCH --error="./logs/100Mpc_%a.err"

readarray -t FILES < ./groups_L0100.txt

module load gcc/9.2.0 openmpi/4.0.2 hdf5/1.12.0 python/3.7.4
source ~/venv/bin/activate

python main.py -gl ${FILES[@]:$SLURM_ARRAY_TASK_ID*8:8} -b "/fred/oz009/clagos/EAGLE/L0100N1504/data/" -o "/fred/oz009/kproctor/L0100N1504/processed/"
