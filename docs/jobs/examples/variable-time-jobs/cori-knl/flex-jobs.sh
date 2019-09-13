#!/bin/bash
#SBATCH -J test 
#SBATCH -q flex 
#SBATCH -C knl 
#SBATCH -N 1
#SBATCH --time=48:00:00      #the max walltime allowed for flex QOS jobs
#SBATCH --time-min=2:00:00   #the minimum amount of time the job should run

#this is an example to run an MPI+OpenMP job: 
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

srun -n8 -c32 --cpu_bind=cores ./a.out

