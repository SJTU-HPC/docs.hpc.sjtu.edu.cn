#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell
#DW jobdw capacity=10GB access_mode=striped type=scratch

srun check-mpi.intel.cori > ${DW_JOB_STRIPED}/output.txt
ls ${DW_JOB_STRIPED}
cat ${DW_JOB_STRIPED}/output.txt
