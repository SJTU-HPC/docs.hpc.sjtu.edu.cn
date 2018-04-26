#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=68
#SBATCH --constraint=knl

srun check-mpi.intel.cori
