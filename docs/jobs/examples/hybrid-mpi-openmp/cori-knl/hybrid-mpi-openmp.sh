#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=272
#SBATCH --constraint=knl

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=68

srun check-hybrid.intel.cori
