#!/bin/bash 
#SBATCH -J vt_vasp 
#SBATCH -q regular 
#SBATCH -C knl 
#SBATCH -N 2 
#SBATCH --time=48:0:00 
#SBATCH --error=vt_vasp%j.err 
#SBATCH --output=vt_vasp%j.out 
#SBATCH --mail-user=elvis@nersc.gov 
# 
#SBATCH --comment=96:00:00 
#SBATCH --time-min=02:0:00 
#SBATCH --signal=B:USR1@300 
#SBATCH --requeue 
#SBATCH --open-mode=append 
  
#user setting 
export OMP_PROC_BIND=true 
export OMP_PLACES=threads 
export OMP_NUM_THREADS=4 
  
#srun must execute in background and catch signal on wait command 
module load vasp/20171017-knl 
srun -n 32 -c16 --cpu_bind=cores vasp_std & 
  
# put any commands that need to run to continue the next job (fragment) here 
ckpt_vasp() { 
set -x 
restarts=`squeue -h -O restartcnt -j $SLURM_JOB_ID` 
echo checkpointing the ${restarts}-th job 
  
#to terminate VASP at the next ionic step 
echo LSTOP = .TRUE. > STOPCAR 

#wait until VASP to complete the current ionic step, write out WAVECAR file and quit 
srun_pid=`ps -fle|grep srun|head -1|awk '{print $4}'` 
echo srun pid is $srun_pid 
wait $srun_pid 
  
#copy CONTCAR to POSCAR 
cp -p CONTCAR POSCAR 
set +x 
} 
  
ckpt_command=ckpt_vasp 
max_timelimit=48:00:00 
ckpt_overhead=300 
  
# requeueing the job if remaining time >0 
. /usr/common/software/variable-time-job/setup.sh 
requeue_job func_trap USR1 
  
wait

