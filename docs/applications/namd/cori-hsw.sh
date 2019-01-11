#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --constraint=haswell
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2

# make sure Haswell environment is loaded
module swap craype-${CRAY_CPU_TARGET} craype-haswell
module load namd
srun namd2 ${INPUT_FILE}
