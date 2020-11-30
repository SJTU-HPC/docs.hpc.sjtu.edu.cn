.. _siesta:

SIESTA
======

简介
----

SIESTA is both a method and its computer program implementation, to
perform efficient electronic structure calculations and ab initio
molecular dynamics simulations of molecules and solids. SIESTA’s
efficiency stems from the use of a basis set of strictly-localized
atomic orbitals. A very important feature of the code is that its
accuracy and cost can be tuned in a wide range, from quick exploratory
calculations to highly accurate simulations matching the quality of
other approaches, such as plane-wave methods.

π 集群上的SIESTA
---------------------

π 集群系统中已经预装 SIESTA (Intel 版本)，可用以下命令加载:

.. code:: bash

   $ module load siesta

π 集群上的Slurm脚本 slurm.test
-----------------------------------

在 cpu 队列上，总共使用 40 核 (n = 40) 
cpu 队列每个节点配有 40核，所以这里使用了 1 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J nechem_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load siesta/4.0.1-intel-19.0.4-impi

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 siesta < input.in

并使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

-  `SIESTA 官网 <http://departments.icmab.es/leem/siesta/>`__
