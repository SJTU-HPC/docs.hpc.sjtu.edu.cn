#!/bin/bash
#SBATCH -J ata
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --comment=96:00:00
#SBATCH --time-min=2:00:00 #the minimum amount of time the job should run
#SBATCH --time=48:00:00
#SBATCH --error=ata-%j.err
#SBATCH --output=ata-%j.out
#SBATCH --mail-user=elvis@nersc.gov
#
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

#timelimit per job, and the amount of time (in seconds) needed for checkpointing (same as in --signal)
max_timelimit=48:00:00
ckpt_overhead=60

#requeueing the job if reamining time >0
module load ata
. $ATA_DIR/etc/ATA_setup.sh
requeue_job func_trap USR1
#

#user setting
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

#srun must execute in background and catch signal on wait command
srun -n 1 -c 48 --cpu_bind=cores ../test.sh &

wait
