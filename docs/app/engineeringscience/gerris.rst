.. _gerris:

Gerris
======

简介
----
Gerris是求解描述流体流动的偏微分方程的软件程序

使用容器运行Gerris
------------------

运行OpenMPI-1.6.5编译的Gerris脚本如下(gerris.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH -J test
   #SBATCH -p cpu
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 3
   #SBATCH --cpus-per-task=40
   #SBATCH --mail-type=end
   #SBATCH --exclusive

   IMAGE_PATH=/lustre/share/img/x86/gerris/gerris-mpi.sif
   module load openmpi/1.6.5-gcc-4.9.2
   mpirun -np 120 singularity exec Gerris3D YOUR_DATA_FILE

运行串行版Gerris脚本如下(gerris.slurm):

.. code:: bash

   #!/bin/bash

   #SBATCH -J test
   #SBATCH -p small
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 1
   #SBATCH --cpus-per-task=1
   #SBATCH --mail-type=end

   IMAGE_PATH=/lustre/share/img/x86/gerris/gerris.sif
   singularity exec  $IMAGE_PATH Gerris3D YOUR_DATA_FILE


使用如下指令提交：

.. code:: bash
   
   $ sbatch gerris.slurm

