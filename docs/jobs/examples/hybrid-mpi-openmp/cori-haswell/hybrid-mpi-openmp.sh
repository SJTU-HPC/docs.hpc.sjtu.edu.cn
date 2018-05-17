#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=16

srun check-hybrid.intel.cori
