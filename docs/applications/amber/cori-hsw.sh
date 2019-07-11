#!/bin/bash -l
#SBATCH -J test_amber
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o test_amber.o%j

module load amber

#Cori has 32 cores per compute node, so run 64 tasks on two nodes (uncomment the following srun command line)
srun -n 64 sander.MPI -i mytest.in -o mytest.out ... (more sander command line options)

