#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=100
#SBATCH --constraint=knl
#SBATCH --ntasks-per-node=66
#SBATCH --cpus-per-task=4
#SBATCH --core-spec=2
#SBATCH --switches=1@20

# make sure KNL environment is loaded
module unload ${CRAY_CPU_TARGET}
module load craype-mic-knl
# make sure correct hugepages module is loaded
module unload $(module -l list 2>&1 | grep craype-hugepages | awk '{print $1}')
module load craype-hugepages8M
module load rca
module load namd
export HUGETLB_DEFAULT_PAGE_SIZE=8M
export HUGETLB_MORECORE=no

srun namd2 ${INPUT_FILE}
