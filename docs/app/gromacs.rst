.. _gromacs:

GROMACS
=======

简介
----

GROMACS
是一种分子动力学应用程序，可以模拟具有数百至数百万个粒子的系统的牛顿运动方程。GROMACS旨在模拟具有许多复杂键合相互作用的生化分子，例如蛋白质，脂质和核酸。

Pi 上的 GROMACS
---------------

Pi 上有多种版本的 GROMACS:

-  |cpu| `CPU版本GROMACS`_

-  |cpu| `CPU版本GROMACS（双精度）`_

-  |gpu| `GPU版本GROMACS`_

-  |gpu| `GPU版本GROMACS+MPI`_

-  |arm| arm

|cpu| (CPU) GROMACS 模块调用
----------------------------------

查看 Pi 上已编译的软件模块:

.. code:: bash

   $ module av gromacs

调用某个版本的 gromacs 模块:

.. code:: bash

   $ module load gromacs/2020-cpu


|cpu| CPU 版本的 GROMACS
------------------------

.. _CPU版本GROMACS:

使用 CPU 版本的 GROMACS
~~~~~~~~~~~~~~~~~~~~~~~

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH -J gromacs_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load gromacs/2020-cpu

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 gmx_mpi mdrun -s ./ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 10000

用下方语句提交作业

.. code:: bash

   $ sbatch slurm.test

.. _CPU版本GROMACS（双精度）:


使用 CPU 版本的 GROMACS (双精度)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH -J gromacs_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load gromacs/2020-cpu-double

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 gmx_mpi_d mdrun -s ./ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 10000

用下方语句提交作业

.. code:: bash

   $ sbatch slurm.test

.. _GPU版本GROMACS:

|gpu| GPU 版本的 GROMACS
------------------------

Pi 集群已预置最新的 GPU GROMACS。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J gromacs_gpu_test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=6
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:1

   module purge
   module load gromacs/2020-gpu

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 gmx mdrun -deffnm benchmark -ntmpi 6 -ntomp 1

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test


.. _GPU版本GROMACS+MPI:

|gpu| GPU 版本的 GROMACS (MPI 版)
---------------------------------

Pi 集群已预置最新的 GPU GROMACS MPI 版。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J gromacs_gpu_test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=12
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:2

   module purge
   module load gromacs/2020-dgx-mpi

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 gmx_mpi mdrun -deffnm benchmark -ntmpi 6 -ntomp 1

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

性能测试
--------

本测试中使用到的测试算例均可在
``/lustre/share/benchmarks/gromacs``\ 找到，用户可自行取用测试。测试时，需将上述目录复制到家目录下。

Gromacs在CPU上的性能测试
^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``ion_channel.tpr`` 算例，不同Gromacs模块在单节点、2节点、4节点性能如下表所示，性能单位为 ``ns/day`` ，越高越好。

+------------+------------+-----------+
|                  | 1 Node  | 2 Nodes  | 3 Nodes    |
+============+============+===========+
| gromacs/2020-cpu | -  | - | - |
+------------+------------+-----------+

测试使用了 GROMACS 提供的 Benchmark 算例进行了 CPU 和 GPU
的性能进行对比。其中 cpu 测试使用单节点40核心，dgx2 测试分配 1 块 gpu
并配比 6 核心。

========================= ===================
Settings                  Performance(ns/day)
========================= ===================
CPU (2019.2-gcc/8.3)      43.718
CPU (2019.2-gcc/9.2)      43.362
CPU (2019.4-gcc/8.3)      43.783
CPU (2019.4-gcc/9.2)      43.057
========================= ===================

Gromacs在CPU上的性能测试
^^^^^^^^^^^^^^^^^^^^^^^^

参考资料
--------

- gromacs官方网站 http://www.gromacs.org/
- NVIDIA GPU CLOUD https://ngc.nvidia.com
- Singularity文档 https://sylabs.io/guides/3.5/user-guide/
