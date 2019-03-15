#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=8
#SBATCH --constraint=haswell

module load impi
mpiicc -qopenmp -o mycode.exe mycode.c

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun -n 32 -c 16 --cpu-bind=cores ./mycode.exe
