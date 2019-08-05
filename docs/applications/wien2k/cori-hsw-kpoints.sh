#!/bin/bash
#SBATCH -n 64 
#SBATCH -t 03:00:00
#SBATCH -q regular
#SBATCH -J test_wien2k 
#SBATCH -C haswell

#module load wien2k
#generating .machines file for k-point and mpi parallel lapw1/2
let ntasks_per_kgroup=1
gen.machines -m $ntasks_per_kgroup

#need to disable SLURM envs hereafter
unset `env|grep SLURM_|awk -F= '{print $1}'`

#put your Wien2k command here
run_lapw -p -i 2 -ec .0001 -I -in1ef

#remove leftover .machines file
rm -fr .machine* 

