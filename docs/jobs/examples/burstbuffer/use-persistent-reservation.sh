#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#DW persistentdw name=PRname

ls $DW_PERSISTENT_STRIPED_PRname/
