#!/bin/bash
#SBATCH -qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --core-spec=2
#SBATCH --tasks-per-node=66
#SBATCH --constraint=knl

module load vasp/5.4.4-knl
srun vasp_std
