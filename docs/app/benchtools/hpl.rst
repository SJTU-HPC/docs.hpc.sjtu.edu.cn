HPL
===

简介
----

+------------------+-------------+
| 模块名           | 平台        |
+==================+=============+
| hpl@2.3          | |cpu| |arm| |
+------------------+-------------+

HPL（The High-Performance Linpack Benchmark）是测试高性能计算集群系统浮点性能的基准程序。HPL通过对高性能计算集群采用高斯消元法求解一元N次稠密线性代数方程组的测试，评价高性能计算集群的浮点计算能力。

浮点计算峰值是指计算机每秒可以完成的浮点计算次数，包括理论浮点峰值和实测浮点峰值。理论浮点峰值是该计算机理论上每秒可以完成的浮点计算次数，主要由CPU的主频决定。理论浮点峰值＝CPU主频×CPU核数×CPU每周期执行浮点运算的次数。本文将为您介绍如何利用HPL测试实测浮点峰值。


使用Intel预编译程序测试HPL性能
------------------------------

Intel HPL使用时建议在每一个NUMA Socket启动一个MPI进程，然后再由MPI进程启动与Socket核心数匹配的计算线程。由于Intel HPL不使用OpenMP库，因此无法通过OMP环境变量控制计算线程数。

运行单节点Intel HPL性能测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~

这个例子以方式在1个双路Intel 6248节点上运行HPL 测试，每个CPU Socket启动1个MPI进程，共启动2个MPI进程。

首先，载入Intel套件模块。

.. code:: bash

   $ module purge; module load intel-parallel-studio/cluster.2020.1

然后，复制Intel HPL算例目录：

.. code:: bash

   $ cp -r $MKLROOT/benchmarks/mp_linpack ./
   $ cd mp_linpack

提高算例输入文件 ``HPL.dat`` 中的问题规模 ``Ns`` 。建议调整至占用整机内存90%左右。这里使用sed将Ns替换为经验值 ``100000`` 。

.. code:: bash

   $ sed -i -e 's/.*Ns.*/100000\ Ns/' HPL.dat

调整HPL.dat的 ``Ps`` 和 ``Qs`` 值，使其乘积等于MPI进程总数。
这里使用sed将 ``Ps`` 和 ``Qs`` 值分别设置为2、1，乘积等于线程总数2。

.. code:: bash

   $ sed -i -e 's/.*\ Ps.*/2\ Ps/' HPL.dat
   $ sed -i -e 's/.*\ Qs.*/1\ Qs/' HPL.dat

编写如下SLURM作业脚本 ``hpl.slurm`` ，使用 ``-n`` 指定MPI进程总数、 ``--ntasks-per-node`` 指定每节点启动的MPI进程数、 ``--cpus-per-task`` 指定每个MPI进程使用的CPU核心数。

.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=hplonenode
    #SBATCH --partition=cpu
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=20
    #SBATCH --exclusive

    ulimit -s unlimited
    ulimit -l unlimited

    module load intel-parallel-studio/cluster.2020.1

    ./runme_intel64_dynamic

使用 ``sbatch hpl.slurm`` 提交后，主要运行结果如下，Intel 6248单节点HPL性能约为1.96Tflops。

.. code:: bash

   ================================================================================
   T/V                N    NB     P     Q               Time                 Gflops
   --------------------------------------------------------------------------------
   WC00C2R100000      100000   192     2     1             339.99            1.96090e+03
   HPL_pdgesv() start time Sun Jan 23 22:00:41 2022


运行多节点Intel HPL性能测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~

该运行实例在2个双路Intel 6248节点上运行HPL 测试，每个CPU Socket启动1个MPI进程，共启动4个MPI进程。

首先，载入Intel套件模块。

.. code:: bash

   $ module purge; module load intel-parallel-studio/cluster.2020.1

然后，复制Intel HPL算例目录

.. code:: bash

   $ cp -r $MKLROOT/benchmarks/mp_linpack ./
   $ cd mp_linpack

计算节点内存为180G，将输入文件 ``HPL.dat`` 中的问题规模 ``Ns`` 调整至内存空间的90%左右。这里使用sed将Ns替换为175718。

.. code:: bash

   $ sed -i -e 's/.*Ns.*/175718\ Ns/' HPL.dat

调整HPL.dat的 ``Ps`` 和 ``Qs`` 值，使其乘积等于MPI进程总数。
这里使用sed将 ``Ps`` 和 ``Qs`` 值分别设置为2、2，乘积等于线程总数2。

.. code:: bash

   $ sed -i -e 's/.*\ Ps.*/2\ Ps/' HPL.dat
   $ sed -i -e 's/.*\ Qs.*/2\ Qs/' HPL.dat

将runme_intel64_dynamic中的MPI总数改为4

.. code:: bash
    
   sed -i 's/MPI_PROC_NUM=2/MPI_PROC_NUM=4/' runme_intel64_dynamic

编写如下SLURM作业脚本 ``hpl.slurm`` ，使用 ``-n`` 指定MPI进程总数， ``--ntasks-per-node`` 指定每节点启动的MPI进程数， ``--cpus-per-task`` 指定每个MPI进程使用的CPU核心数。

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
   
   module load intel-parallel-studio/cluster.2020.1
   
   ./runme_intel64_dynamic

使用 ``sbatch hpl.slurm`` 提交后，主要运行结果如下，Intel 6248双节点HPL性能约为4.35Tflops。

.. code:: bash

   ================================================================================
   T/V                N    NB     P     Q               Time                 Gflops
   --------------------------------------------------------------------------------
   WR00C2R2      175718   256     2     2             830.63            4.35466e+03

ARM平台测试HPL性能
------------------

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


参考资料
--------

- Running the Intel Distribution for LINPACK Benchmark https://www.intel.com/content/www/us/en/develop/documentation/onemkl-linux-developer-guide/top/intel-oneapi-math-kernel-library-benchmarks/intel-distribution-for-linpack-benchmark-1/run-the-intel-distribution-for-linpack-benchmark.html
- HOW DO I TUNE MY HPL.DAT FILE? https://www.advancedclustering.com/act_kb/tune-hpl-dat-file/
