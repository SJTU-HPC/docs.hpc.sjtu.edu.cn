#!/bin/bash

#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load gromacs/2019.4-gcc-9.2.0-openmpi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 gmx_mpi mdrun -s ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 10000

