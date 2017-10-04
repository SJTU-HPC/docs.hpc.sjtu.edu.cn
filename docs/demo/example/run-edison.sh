#!/bin/bash
#SBATCH -N 2
#SBATCH -p debug
#SBATCH -t 00:05:00

#OpenMP settings:
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
sbcast ./xthi /tmp/xthi
srun -n 12 -c 8 --cpu_bind=cores /tmp/xthi | sort
