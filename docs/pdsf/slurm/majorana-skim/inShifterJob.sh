#!/bin/bash
# Here is a minimally working example. I set it up s/t I can pass arbitrary one-line commands to it, e.g.:  
#  sbatch batchShifter.slr './skim_mjd_data 1 1'
#  sbatch batchShifter.slr 'echo ABC'
echo "Job Start:"
date
echo "Node(s): "$SLURM_JOB_NODELIST
echo "Job ID: "$SLURM_JOB_ID
if [ -n "$SHIFTER_RUNTIME" ]; then
    echo "Shifter image active."
fi
# This runs whatever commands we pass to it.
echo "${@}"
time ${@}
echo "Job Complete:"
date
