#!/bin/bash
#SBATCH -qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

module load vasp
srun vasp_std
