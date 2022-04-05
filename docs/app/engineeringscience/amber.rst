.. _amber:

amber
======

简介
----

Amber 是分子动力学软件，用于蛋白质、核酸、糖等生物大分子的计算模拟。Amber 为商业软件，需购买授权使用。

如需使用集群上的 Amber，请发邮件至 hpc 邮箱，附上课题组购买 Amber 的证明，并抄送超算帐号负责人。


ARM 版 AMBER
-------------

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
