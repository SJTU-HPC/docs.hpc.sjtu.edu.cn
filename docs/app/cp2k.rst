CP2k
====

CP2K is a quantum chemistry and solid state physics software package
that can perform atomistic simulations of solid state, liquid,
molecular, periodic, material, crystal, and biological systems.

Pi上的CP2K
------------

Pi2.0 系统中已经预装 CP2K (GNU+cpu 版本)，可用以下命令加载:

.. code:: bash

   $ module load cp2k/6.1-gcc-8.3.0-openblas-openmpi

Pi上的Slurm脚本slurm.test
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
   module load cp2k/6.1-gcc-8.3.0-openblas-openmpi

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 cp2k.popt -i example.inp

并使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

-  `CP2K 官网 <https://manual.cp2k.org/#gsc.tab=0>`__
