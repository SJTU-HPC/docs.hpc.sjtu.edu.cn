#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=8
#SBATCH --constraint=knl
#SBATCH --ntasks-per-node=68
#SBATCH --cpus-per-task=4

# make sure KNL environment is loaded
module swap craype-${CRAY_CPU_TARGET} craype-mic-knl
module load namd
srun namd2 ${INPUT_FILE}
