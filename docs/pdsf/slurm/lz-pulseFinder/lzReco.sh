#!/bin/bash
if [[ $SLURM_JOB_PARTITION == *"-chos" ]]
then
   echo  task-in-chos
   chosenv
else
  echo  task-in-shifter
  #source /usr/share/Modules/init/bash
  #module use /usr/common/usg/Modules/modulefiles
  echo inShifter:`env|grep  SHIFTER_RUNTIME`
  cat /etc/*release

  export CHOS=sl64
  source ~/.bash_profile.ext
fi

echo check if input exist
ls -l /project/projectdirs/lz/data/simulations/LUXSim_release-4.4.6_geant4.9.5.p02/full_slow_simulation/electron_recoils/FullSlowSimulation_ER_flat_10k_DER.root 

source /global/project/projectdirs/lz/releases/physics/latest/Physics/setup.sh
lzap_project
echo run the task for 6-7 minutes
time lzap scripts/validations/PulseFinderValidation.py
echo " lzReco completed on "`date`