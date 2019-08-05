#!/bin/bash
#SBATCH –N 1 
#SBATCH -C haswell
#SBATCH –q regular
#SBATCH –t 6:00:00
 
module load vasp
srun –n32 –c2 --cpu_bind=cores vasp_std

