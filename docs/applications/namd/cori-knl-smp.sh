#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=8
#SBATCH --constraint=knl
#SBATCH --ntasks-per-node=34
#SBATCH --cpus-per-task=8

# make sure KNL environment is loaded
module swap craype-${CRAY_CPU_TARGET} craype-mic-knl
module load namd/2.13-smp
srun --cpu-bind=cores namd2 ++ppn 2 ${INPUT_FILE}
