HPL
===

简介
----

HPL（The High-Performance Linpack Benchmark）是测试高性能计算集群系统浮点性能的基准程序。HPL通过对高性能计算集群采用高斯消元法求解一元N次稠密线性代数方程组的测试，评价高性能计算集群的浮点计算能力。

浮点计算峰值是指计算机每秒可以完成的浮点计算次数，包括理论浮点峰值和实测浮点峰值。理论浮点峰值是该计算机理论上每秒可以完成的浮点计算次数，主要由CPU的主频决定。理论浮点峰值＝CPU主频×CPU核数×CPU每周期执行浮点运算的次数。本文将为您介绍如何利用HPL测试实测浮点峰值。

导入HPL环境
-----------

+--------+-------+----------+--------------------------------------------+
| 版本   | 平台  | 构建方式 | 模块名                                     |
+========+=======+==========+============================================+
| 2.3    | |cpu| | spack    | oneapi/2021.4.0  思源一号                  |
+--------+-------+----------+--------------------------------------------+
| 2.3    | |cpu| | spack    | oneapi/2021.4.0  π2.0                      |
+--------+-------+----------+--------------------------------------------+
| 2.3    | |cpu| | spack    | hpl/2.3-gcc-9.3.0-openblas-openmpi ARM集群 |
+--------+-------+----------+--------------------------------------------+

.. code:: bash

   mkdir ~/HPL && cd ~/HPL
   module load oneapi/2021.4.0
   cp -r $MKLROOT/benchmarks/mp_linpack ./
   cd mp_linpack/

文件目录结构如下所示：

.. code:: bash

   [hpc@login3 80cores]$ tree mp_linpack/
   mp_linpack/
   ├── build.sh
   ├── COPYRIGHT
   ├── HPL.dat
   ├── HPL_main.c
   ├── libhpl_intel64.a
   ├── readme.txt
   ├── runme_intel64_dynamic
   ├── runme_intel64_prv
   ├── run.slurm
   ├── xhpl_intel64_dynamic
   └── xhpl_intel64_dynamic_outputs.txt

测试平台
--------

- `π2.0`_

- `思源一号平台`_

- `ARM平台`_
  
.. _π2.0:

π2.0
----

Intel HPL使用时建议在每一个NUMA Socket启动一个MPI进程，然后再由MPI进程启动与Socket核心数匹配的计算线程。由于Intel HPL不使用OpenMP库，因此无法通过OMP环境变量控制计算线程数。

π2.0上计算节点配置信息：双路Intel 6248节点，每个CPU Socket启动1个MPI进程，共启动2个MPI进程。

首先需要配置文件内容：

计算节点内存为180G，将输入文件 ``HPL.dat`` 中的问题规模 ``Ns`` 调整至内存空间的80%左右 ``0.8*sqrt(mem*1024*1024*1024*nodes/8)`` 。本算例使用了两个节点，这里使用sed>将Ns替换为176640。

.. code:: bash

   $ sed -i -e 's/.*Ns.*/176640\ Ns/' HPL.dat

然后调整HPL.dat的 ``Ps`` 和 ``Qs`` 值，使其乘积等于MPI进程总数。
这里使用sed将 ``Ps`` 和 ``Qs`` 值分别设置为2、2，乘积等于线程总数2。

.. code:: bash

   $ sed -i -e 's/.*\ Ps.*/2\ Ps/' HPL.dat
   $ sed -i -e 's/.*\ Qs.*/2\ Qs/' HPL.dat

接下来将runme_intel64_dynamic中的MPI总数改为4

.. code:: bash

   sed -i 's/MPI_PROC_NUM=2/MPI_PROC_NUM=4/' runme_intel64_dynamic

提交如下运行脚本：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=hpl2node
   #SBATCH --partition=cpu
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 4
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=20
   #SBATCH --exclusive
   
   ulimit -s unlimited
   ulimit -l unlimited
   
   module load oneapi
   
   ./runme_intel64_dynamic

使用 ``-n`` 指定MPI进程总数， ``--ntasks-per-node`` 指定每节点启动的MPI进程数， ``--cpus-per-task`` 指定每个MPI进程使用的CPU核心数

使用如下命令提交脚本：

.. code:: bash

   sbatch run.slurm

运行结果如下所示：

.. code:: bash

   ================================================================================
   T/V                N    NB     P     Q               Time                 Gflops
   --------------------------------------------------------------------------------
   WC00C2R100000      176640   256     2     2             973.53            3.77426e+03

.. _思源一号平台:

思源一号
--------

文件参数的配置，参考上述规则，将 ``N`` 、 ``P`` 、 ``Q`` 等参数使用sed命令更改如下：

.. code:: bash

   sed -i -e 's/.*Ns.*/209510\ Ns/' HPL.dat
   sed -i -e 's/.*\ Ps.*/2\ Ps/' HPL.dat
   sed -i -e 's/.*\ Qs.*/2\ Qs/' HPL.dat
   sed -i 's/MPI_PROC_NUM=2/MPI_PROC_NUM=4/' runme_intel64_dynamic

运行脚本如下所示：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=hpl2node
   #SBATCH --partition=64c256g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 4
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=32
   #SBATCH --exclusive
   
   ulimit -s unlimited
   ulimit -l unlimited
   
   module load oneapi
   
   ./runme_intel64_dynamic

.. _ARM平台:

ARM集群
-------

首先，复制算例到本地。

.. code:: bash

   $ mkdir arm_hpl
   $ cd arm_hpl
   $ cp -r /lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-9.3.0/hpl-2.3-svu3iccgwr6whf7b2fcj7mbkaipbffye/bin/* ./

然后，将输入文件 ``HPL.dat`` 中的问题规模 ``Ns`` 调整至内存空间256G的80%左右。
这里使用sed将Ns替换为147840。

.. code:: bash

   $ sed -i -e 's/.*Ns.*/147840\ Ns/' HPL.dat

将 ``NB`` 更改为经验值384。

.. code:: bash

   $ sed -i -e 's/.*NBs.*/384\ NBs/' HPL.dat

接下来，将将 ``Ps`` 和 ``Qs`` 值分别设置为8、16，乘积等于CPU总核数128。

.. code:: bash

   $ sed -i -e 's/.*\ Ps.*/8\ Ps/' HPL.dat
   $ sed -i -e 's/.*\ Qs.*/16\ Qs/' HPL.dat

使用 ``sbatch hpl.slurm`` 提交作业，其中 ``N`` 代表节点总数， ``ntasks-per-node`` 代表每个节点使用的总核数。

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=arm_hpl       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=128
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
    
   export OMP_NUM_THREADS=1
   module load openmpi/4.0.3-gcc-9.2.0
   module load hpl/2.3-gcc-9.3.0-openblas-openmpi
   ulimit -s unlimited
   ulimit -l unlimited
   mpirun -np $SLURM_NTASKS xhpl

运行结果如下所示：

.. code:: bash

   ================================================================================
   T/V                N    NB     P     Q               Time                 Gflops
   --------------------------------------------------------------------------------
   WR00L2L2      147840   384     8    16            2489.13             8.6545e+02

运行结果时间比较
----------------

π2.0集群
~~~~~~~~

+-----------------------------------------+
|             oneapi/2021.4.0             |
+===========+=========+=========+=========+
| 核数      | 40      | 80      | 160     |
+-----------+---------+---------+---------+
| Time(s)   | 705.40  | 973.53  | 1439.61 |
+-----------+---------+---------+---------+
| Gflops    | 1847.25 | 3774.26 | 7117.28 |
+-----------+---------+---------+---------+

思源一号集群
~~~~~~~~~~~~

+-----------------------------------------+
|            oneapi/2021.4.0              |
+===========+=========+=========+=========+
| 核数      | 64      | 128     | 256     |
+-----------+---------+---------+---------+
| Time(s)   | 1548.69 | 2247.28 | 3111.47 |
+-----------+---------+---------+---------+
| Gflops    | 3958.81 | 7728.58 | 15798.2 |
+-----------+---------+---------+---------+

参考资料
--------

- Running the Intel Distribution for LINPACK Benchmark https://www.intel.com/content/www/us/en/develop/documentation/onemkl-linux-developer-guide/top/intel-oneapi-math-kernel-library-benchmarks/intel-distribution-for-linpack-benchmark-1/run-the-intel-distribution-for-linpack-benchmark.html
- HOW DO I TUNE MY HPL.DAT FILE? https://www.advancedclustering.com/act_kb/tune-hpl-dat-file/
