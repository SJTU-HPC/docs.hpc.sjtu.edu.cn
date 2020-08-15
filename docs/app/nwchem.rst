#

.. raw:: html

   <center>

NWChem

.. raw:: html

   </center>

--------------

简介
----

NWChem aims to provide its users with computational chemistry tools that
are scalable both in their ability to treat large scientific
computational chemistry problems efficiently, and in their use of
available parallel computing resources from high-performance parallel
supercomputers to conventional workstation clusters.

Pi 上的 NWChem
--------------

Pi2.0 系统中已经预装 NWChem-6.8.1 (GNU+cpu 版本)，可用以下命令加载:

.. code:: bash

   $ module load nwchem/6.8.1-gcc-8.3.0-openblas-openmpi

Pi 上的 Slurm 脚本 slurm.test
-----------------------------

在 cpu 队列上，总共使用 40 核 (n = 40) cpu 队列每个节点配有 40
核，所以这里使用了 1 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J nechem_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load nwchem/6.8.1-gcc-8.3.0-openblas-openmpi

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 nwchem

并使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

参考链接
--------

-  `NWChem 官网 <https://nwchemgit.github.io/>`__
