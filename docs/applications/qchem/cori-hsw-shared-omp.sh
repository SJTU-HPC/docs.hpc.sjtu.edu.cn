#!/bin/bash
#SBATCH -q shared
#SBATCH -n 2
#SBATCH -t 1:00:00
#SBATCH -J my_job
#SBATCH -L SCRATCH

module load qchem
qchem -slurm -nt 2 B3LYP_water.in 

