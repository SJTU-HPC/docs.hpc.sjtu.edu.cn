#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#DW jobdw capacity=10GB access_mode=striped type=scratch
#DW stage_out source=$DW_JOB_STRIPED/output destination=/global/cscratch1/sd/username/output type=directory
mkdir $DW_JOB_STRIPED/output
srun check-mpi.intel.cori > ${DW_JOB_STRIPED}/output/output.txt
