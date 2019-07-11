#!/bin/bash -l
#SBATCH -J test_molpro
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o test_molpro.o%j

module load molpro
molpro -n 32 h20_opt_dflmp2.test

