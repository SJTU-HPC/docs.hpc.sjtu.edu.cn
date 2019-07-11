#!/bin/bash -l
#SBATCH -J test_molpro
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 06:00:00
#SBATCH -o test_molpro.o%j

module load molpro
molpro -n 8 pentane_dflccsd.test

