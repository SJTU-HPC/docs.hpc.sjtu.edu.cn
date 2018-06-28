#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --time=2
#SBATCH --array=0-2

echo $SLURM_ARRAY_TASK_ID
