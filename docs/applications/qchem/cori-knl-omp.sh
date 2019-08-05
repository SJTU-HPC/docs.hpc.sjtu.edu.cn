#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J my_job
#SBATCH -L SCRATCH
#SBATCH -C knl 

module load qchem
qchem -slurm -nt 68 B3LYP_water.in 

