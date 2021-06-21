.. _gerris:

Gerris
======

简介
----
Gerris是求解描述流体流动的偏微分方程的软件程序

使用容器运行Gerris
------------------

运行脚本如下(gerris.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH -J test
   #SBATCH -p cpu
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 3
   #SBATCH --cpus-per-task=40
   #SBATCH --mail-type=end
   #SBATCH --mail-user=********@163.com
   #SBATCH --exclusive
   IMAGE_PATH=/lustre/home/acct-hpc/hpchgc/Gerris/test_install_in_singulatiry/gerris.sif
   export PATH=$PATH:/lustre/home/acct-hpc/hpchgc/Gerris/openmpi-1.6.5-install/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/home/acct-hpc/hpchgc/Gerris/openmpi-1.6.5-install/lib
   export MPI_DIR="/lustre/home/acct-hpc/hpchgc/Gerris/openmpi-1.6.5-install"
   mpirun -np 120 singularity exec -B /lustre/home/acct-hpc/hpchgc/Gerris/test_gerris_u2cb/test3:/mnt  $IMAGE_PATH /mnt/./run.sh

使用如下指令提交：

.. code:: bash
   
   $ sbatch gerris.slurm

gerris脚本中run.sh内容为(testb.gfs为运行的数据)
-----------------------------------------------

.. code:: bash
      
   #!/bin/bash

   gerris2D testb.gfs
