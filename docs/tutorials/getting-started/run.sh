#!/bin/bash
#SBATCH -N 2
#SBATCH -p debug
#SBATCH -t 00:05:00

srun -n 48 ./hello.x | sort -n
