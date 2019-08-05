#!/bin/bash
#SBATCH –N 2 
#SBATCH -C knl
#SBATCH –q regular
#SBATCH –t 6:00:00

module load vasp/20181030-knl
export OMP_NUM_THREADS=4

# launching 1 task every 4 cores (16 CPUs)
srun –n32 –c16 --cpu_bind=cores vasp_std

