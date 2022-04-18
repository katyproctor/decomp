#!/bin/bash 
#SBATCH --job-name=25Mpc_hr
#SBATCH --time=04:20:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --array=2-22

#SBATCH --output="./logs/25Mpc_%a.out"
#SBATCH --error="./logs/25Mpc_%a.err"

readarray -t FILES < ./groups_L0025_hr_ref.txt

module load gcc/9.2.0 openmpi/4.0.2 hdf5/1.12.0 python/3.7.4
source ~/venv/bin/activate

python main.py -gl ${FILES[@]:$SLURM_ARRAY_TASK_ID*25:25} -b "/fred/oz009/clagos/EAGLE/L0025N0752/PE/REFERENCE/data/" -o "/fred/oz009/kproctor/L0025N0752_Ref/processed/"
