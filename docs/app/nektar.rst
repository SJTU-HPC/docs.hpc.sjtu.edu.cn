.. _nektar:

Nektar++
==========

简介
----

Nektar++ is a spectral/hp element framework designed to support the
construction of efficient high-performance scalable solvers for a wide
range of partial differential equations.

Pi上的Nektar++
----------------

查看 Pi 上已编译的软件模块:

.. code:: bash

   $ module avail Nektar

加载预安装的Nektar++
---------------------

Pi 2.0 系统中已经预装 nektar-5.0.0 (intel 版本)，可以用以下命令加载:

::

   $ module load nektar/5.0.0-intel-19.0.4-impi

提交Intel版本Nektar++作业
-----------------------------

使用 intel 编译的 CPU 版本 Nektar 运行单节点作业脚本示例
nektar_cpu_intel.slurm 如下：

.. code:: bash

   #!/bin/bash

   #SBATCH -J Nektar_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load nektar/5.0.0-intel-19.0.4-impi

   ulimit -s unlimited
   ulimit -l unlimited

   srun IncNavierStokesSolver-rg

并使用如下指令提交：

.. code:: bash

   $ sbatch nektar_cpu_intel.slurm

参考资料
--------

-  `Nektar 官网 <https://www.nektar.info/>`__
