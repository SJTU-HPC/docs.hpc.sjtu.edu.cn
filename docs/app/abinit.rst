.. _abinit:

ABINIT
======

简介
----

ABINIT is a DFT code based on pseudopotentials and a planewave basis,
which calculates the total energy, charge density and electronic
structure for molecules and periodic solids. In addition to many other
features, it provides the time dependent DFT, or many-body perturbation
theory (GW approximation) to compute the excited states.

π 集群上的 ABINIT
------------

查看 π 集群上已编译的软件模块:

.. code:: bash

   $ module avail abinit

调用该模块:

.. code:: bash

   $ module load abinit/8.10.3-gcc-9.2.0-openblas-openmpi

π 集群上的 Slurm 脚本 slurm.test
-----------------------------

在 cpu 队列上，总共使用 80 核 (n = 80)
cpu 队列每个节点配有 40核，所以这里使用了 2 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J abinit_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load abinit

   srun --mpi=pmi2 abinit < example.in

π 集群上提交作业
-------------

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

-  ABINIT http://www.abinit.org
