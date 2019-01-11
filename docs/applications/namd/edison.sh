#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=2

# make sure correct hugepages module is loaded
module unload $(module -l list 2>&1 | grep craype-hugepages | awk '{print $1}')
module load craype-hugepages8M
module load rca
module load namd
export HUGETLB_DEFAULT_PAGE_SIZE=8M
export HUGETLB_MORECORE=no

srun namd2 ${INPUT_FILE}
