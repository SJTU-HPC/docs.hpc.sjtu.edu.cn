#!/bin/bash
#SBATCH -N 4
#SBATCH -n 128
#SBATCH -t 03:00:00
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -J test_wien2k 

#module load wien2k
#generating .machines file for k-point and mpi parallel lapw1/2
let ntasks_per_kgroup=8
gen.machines -m $ntasks_per_kgroup

#need to disable SLURM envs hereafter
unset `env|grep SLURM_|awk -F= '{print $1}'`

#put your Wien2k command here
run_lapw -p -i 2 -ec .0001 -I -in1ef 

#remove leftover .machines file
rm -fr .machine*

