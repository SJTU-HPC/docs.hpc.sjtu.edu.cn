.. _amber:

amber
=====

简介
----

amber

ARM版AMBER
----------

ARM平台上运行脚本如下(amber.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 2          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   source /lustre/share/singularity/commercial-app/amber/activate arm

   mpirun -n $SLURM_NTASKS pmemd.MPI ...

使用如下指令提交：

.. code:: bash

   $ sbatch amber.slurm


思源平台Amber
---------------

思源平台上运行脚本如下(amber.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=64c512g    
   #SBATCH -N 2          
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH --exclusive

   source /dssg/share/imgs/commercial-app/amber/activate 18cpu

   mpirun -n $SLURM_NTASKS pmemd.MPI ...

使用如下指令提交：

.. code:: bash

   $ sbatch amber.slurm
