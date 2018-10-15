#!/bin/bash -l

#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --time=10:00
#SBATCH --licenses=project,cscratch1
#SBATCH --constraint=haswell

srun -n 128 -c 2 --cpu_bind=cores ./a.out   
srun -n 64 -c 4 --cpu_bind=cores ./b.out 
srun -n 32 -c 8 --cpu_bind=cores ./c.out
