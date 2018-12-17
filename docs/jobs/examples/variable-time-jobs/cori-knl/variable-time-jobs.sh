#!/bin/bash
#SBATCH -J vtj
#SBATCH -q regular
#SBATCH -C knl 
#SBATCH -N 2
#SBATCH --comment=96:00:00
#SBATCH --time-min=2:00:00 #the minimum amount of time the job should run
#SBATCH --time=48:00:00
#SBATCH --error=vtj-%j.err
#SBATCH --output=vtj-%j.out
#SBATCH --mail-user=elvis@nersc.gov
#
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

# use the following three variables to specify the time limit per job (max_timelimit), 
# the amount of time (in seconds) needed for checkpointing, 
# and the command to use to do the checkpointing if any (leave blank if none)
max_timelimit=48:00:00   # can match the #SBATCH --time option but don't have to
ckpt_overhead=60         # should match the time in the #SBATCH --signal option
ckpt_command=

# requeueing the job if reamining time >0 (do not change the following 3 lines )
. /usr/common/software/variable-time-job/setup.sh
requeue_job func_trap USR1
#

# user setting goes here
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

#srun must execute in the background and catch the signal USR1 on the wait command
srun -n32 -c16 --cpu_bind=cores ./a.out &

wait

