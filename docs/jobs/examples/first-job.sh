#!/bin/bash -l
#SBATCH --nodes=2         
#SBATCH --time=00:10:00   
#SBATCH --constraint=knl  
#SBATCH --license=SCRATCH 
#SBATCH --qos=regular     

export RUNDIR=$SCRATCH/run-$SLURM_JOBID
mkdir -p $RUNDIR
cd $RUNDIR

srun -n 4 bash -c 'echo "Hello, world, from node $(hostname)"' 

