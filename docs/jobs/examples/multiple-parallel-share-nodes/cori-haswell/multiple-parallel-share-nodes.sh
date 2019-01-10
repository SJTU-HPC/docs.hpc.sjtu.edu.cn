#!/bin/bash

#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 12:00:00
#SBATCH --gres=craynetwork:2
#SBATCH -L SCRATCH
#SBATCH -C haswell

srun -N 2 -n 16 -c 2 --mem=51200 --gres=craynetwork:1 ./exec_a &
srun -N 2 -n 48 -c 2 --mem=61440 --gres=craynetwork:1 ./exec_b &
wait 
