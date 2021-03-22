.. _gromacs:

GROMACS
=======

简介
----

GROMACS
是一种分子动力学应用程序，可以模拟具有数百至数百万个粒子的系统的牛顿运动方程。GROMACS旨在模拟具有许多复杂键合相互作用的生化分子，例如蛋白质，脂质和核酸。

调用GROMACS模块
---------------

超算平台提供了多个预编译的Gromacs模块，可用 ``module load`` 指令加载特定模块。

+----------------------------------+-------+----------+-----------+
| 模块名                           | 平台  | 容器部署 | 单/双精度 |
+==================================+=======+==========+===========+
| gromacs/2020-cpu                 | |cpu| | 是       | 单精度    |
+----------------------------------+-------+----------+-----------+
| gromacs/2020-cpu-double          | |cpu| | 是       | 双精度    |
+----------------------------------+-------+----------+-----------+
| gromacs/2020-dgx                 | |gpu| | 是       | 单精度    |
+----------------------------------+-------+----------+-----------+
| gromacs/2019.2-gcc-9.2.0-openmpi | |cpu| | 否       | 单精度    |
+----------------------------------+-------+----------+-----------+

调用某个版本的 gromacs 模块:

.. code:: bash

   $ module load gromacs/2020-cpu


使用CPU版Gromacs
----------------

CPU版GROMACS作业示例
^^^^^^^^^^^^^^^^^^^^

在cpu队列上，总共使用40核(n = 40)。cpu 队列每个节点配有40核，所以这里使用了1个节点。脚本名称可设为 slurm.test
这个作业脚本申请了40个CPU计算核心，由于 `cpu` 队列上每个节点上有40个计算核心，因此这是一个单节点Gromacs作业。

.. code:: bash

    #!/bin/bash

    #SBATCH -J gromacs_cpu_test
    #SBATCH -p cpu
    #SBATCH -n 40
    #SBATCH --ntasks-per-node=40
    #SBATCH -o %j.out
    #SBATCH -e %j.err

    module load gromacs/2019.4-gcc-9.2.0-openmpi

    ulimit -s unlimited
    ulimit -l unlimited

    srun --mpi=pmi2 gmx_mpi mdrun -s ./ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 10000
    
将 ``/lustre/share/benchmarks/gromacs`` 路径下的 ``ion_channel.tpr`` 文件拷贝到本地：

.. code:: bash

    $ cp /lustre/share/benchmarks/gromacs/ion_channel.tpr .
    

提交作业。

.. code:: bash

   $ sbatch slurm.test

CPU版GROMACS作业示例(双精度)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH -J gromacs__cpu_double_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load gromacs/2020-cpu-double

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 gmx_mpi_d mdrun -s ./ion_channel.tpr -maxh 0.50 -resethway -noconfout -nsteps 10000

将 ``/lustre/share/benchmarks/gromacs`` 路径下的 ``ion_channel.tpr`` 文件拷贝到本地：

.. code:: bash

    $ cp /lustre/share/benchmarks/gromacs/ion_channel.tpr .
    
用下方语句提交作业

.. code:: bash

   $ sbatch slurm.test

.. _GPU版本GROMACS:


GPU版Gromacs(MPI版)
-------------------

π 集群已预置最新的 GPU GROMACS MPI 版。脚本名称可设为 slurm.test

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

   module load gromacs/2020-dgx-mpi

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 gmx_mpi mdrun -deffnm benchmark -ntomp 1 -s ./ion_channel.tpr

将 ``/lustre/share/benchmarks/gromacs`` 路径下的 ``ion_channel.tpr`` 文件拷贝到本地：

.. code:: bash

    $ cp /lustre/share/benchmarks/gromacs/ion_channel.tpr .
    
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

+----------------------------------+------------+------------+-----------+
| 模块                             | 1节点性能  | 2节点性能  | 4节点性能 |
+==================================+============+============+===========+
| gromacs/2020-cpu                 | 43.286     | 71.488     | 118.507   |
+----------------------------------+------------+------------+-----------+
| gromacs/2020.2-gcc-9.2.0-openmpi | 43.491     | 71.401     | 115.569   |
+----------------------------------+------------+------------+-----------+
| gromacs/2019.2-gcc-9.2.0-openmpi | 42.874     | 68.497     | 115.347   |
+----------------------------------+------------+------------+-----------+

Gromacs在GPU上的性能测试
^^^^^^^^^^^^^^^^^^^^^^^^

参考资料
--------

- gromacs官方网站 http://www.gromacs.org/

- Singularity文档 https://sylabs.io/guides/3.5/user-guide/
