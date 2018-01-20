#!/bin/bash
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 100

# make sure correct hugepages module is loaded
module unload $(module -l list 2>&1 | grep craype-hugepages | awk '{print $1}')
module load craype-hugepages8M
module load rca
module load namd
export HUGETLB_DEFAULT_PAGE_SIZE=8M
export HUGETLB_MORECORE=no

srun -n $((SLURM_NNODES*24)) -c 2 namd2 ${INPUT_FILE} 
