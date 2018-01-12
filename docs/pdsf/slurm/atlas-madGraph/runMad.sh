#!/bin/bash -l

IDX=${SLURM_ARRAY_TASK_ID}
if [[ $SLURM_JOB_PARTITION == *"-chos" ]]
then
   echo  task-in-chos
   chosenv
   ls -l /proc/chos/link
else
  echo  task-in-shifter
  echo inShifter:`env|grep  SHIFTER_RUNTIME`
  cat /etc/*release
fi

# increase the number of user processes allowed by a user on a system.
ulimit -s unlimited
ulimit -u 8192
ulimit -a

echo run on node `hostname`
module load python/2.7.9
module list
python -V

printf -v runidx "%05d" ${IDX}

MGDIR=${DATA_STORE-fixMe1}
echo "WORKDIR=${WORKDIR}"
echo "MGDIR=${MGDIR}"

mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo Prepare a copy so MG can edit it
time ( cp /global/project/projectdirs/atlas/kkrizka/PROC_xia.tgz .; tar -zxf PROC_xia.tgz )
ls -l  .

(sleep 120;  echo Dump information after delay; date; top ibn1 ; free -g)&
echo "launch --nb_core=$SLURM_CPUS_ON_NODE" >runX.cmd 
echo "set nevents $NUM_EVE" >>runX.cmd 

echo # Generate a new run with
cat runX.cmd

head -n40 PROC_xia/Cards/run_card.dat
/usr/bin/time -v  ./PROC_xia/bin/madevent runX.cmd

# Save the output
OUTDIR=${MGDIR}/PROC_xia/Events/${SLURM_JOBID}
mkdir -p ${OUTDIR}
cp -r PROC_xia/Events/run_01 ${OUTDIR}/run_${runidx}
echo "Events copied to ${OUTDIR}/run_${runidx}"
echo "task-done  "`date`
