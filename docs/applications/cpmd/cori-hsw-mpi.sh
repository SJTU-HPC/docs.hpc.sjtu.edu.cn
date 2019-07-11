#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH -J my_job
#SBATCH -L SCRATCH
#SBATCH -C haswell

module load qchem
qchem -slurm -mpi -np 2 -nt 32 B3LYP_water.in 

