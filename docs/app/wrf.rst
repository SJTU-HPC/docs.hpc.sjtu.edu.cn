.. _wrf:

WRF
===

简介
----
WRF是

ARM版WRF
--------

ARM平台上运行脚本如下(wrf.slurm):

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1           
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   module load wrf/4.2-gcc-9.3.0-openmpi
   module load openmpi/4.0.3-gcc-9.3.0

   export WRF_HOME=/lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-9.3.0/wrf-4.2-dvii6gqnopsssz5yytk4xcgrk2g2d2ob
   export PBV=CLOSE
   export OMP_PROC_BIND=TRUE
   export OMP_NUM_THREADS=4
   export OMP_STACKSIZE="16M"
   export WRF_NUM_TILES=128

   mpirun -np 32 --bind-to core --map-by ppr:32:node:pe=4 numactl -l $WRF_HOME/main/wrf.exe

使用如下指令提交：

.. code:: bash

   $ sbatch wrf.slurm
