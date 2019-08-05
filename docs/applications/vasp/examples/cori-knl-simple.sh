#!/bin/bash
#SBATCH –N 2 
#SBATCH -C knl
#SBATCH –q regular
#SBATCH –t 6:00:00

module load vasp/5.4.4-knl
srun –n128 -c4 --cpu_bind=cores vasp_std

