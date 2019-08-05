#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --time=30:00
#SBATCH --licenses=cscratch1
#SBATCH --constraint=haswell

srun -N 2 -n 44 -c 2 --cpu_bind=cores ./a.out &
srun -N 4 -n 108 -c 2 --cpu_bind=cores ./b.out &
srun -N 2 -n 40 -c 2 --cpu_bind=cores ./c.out &
wait

