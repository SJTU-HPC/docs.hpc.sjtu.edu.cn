#!/bin/bash
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 04:00:00
#SBATCH -J my_job
#SBATCH -o my_job.o%j
#SBATCH -C haswell

module load cpmd

#There are 32 cores per Cor Haswell node
srun -n 64 cpmd.x test.in [PP-path] > test.out
