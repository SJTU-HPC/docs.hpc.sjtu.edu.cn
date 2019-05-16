#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=4
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

srun --cpu-bind=cores check-hybrid.intel.cori
