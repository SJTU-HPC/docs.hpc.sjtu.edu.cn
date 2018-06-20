#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --constraint=haswell
#DW jobdw capacity=10GB access_mode=striped type=scratch
#DW stage_in source=/global/cscratch1/sd/dwtest-file destination=$DW_JOB_STRIPED/dwtest-file type=file
srun ls ${DW_JOB_STRIPED}/dwtest-file
