#!/bin/bash -l
#SBATCH –N 2 
#SBATCH -C haswell
#SBATCH –q regular
#SBATCH –t 6:00:00
 
module load vasp/20181030-hsw
export OMP_NUM_THREADS=4

# launching 1 task every 4 cores (8 CPUs)
srun –n16 –c8 --cpu_bind=cores vasp_std

