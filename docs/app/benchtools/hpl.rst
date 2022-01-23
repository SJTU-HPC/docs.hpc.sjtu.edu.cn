HPL
===

简介
----

+------------------+-------------+
| 模块名           | 平台        |
+==================+=============+
| hpl@2.3          | |cpu| |arm| |
+------------------+-------------+

HPL（The High-Performance Linpack Benchmark）是测试高性能计算集群系统浮点性能的基准。HPL通过对高性能计算集群采用高斯消元法求解一元N次稠密线性代数方程组的测试，评价高性能计算集群的浮点计算能力。

浮点计算峰值是指计算机每秒可以完成的浮点计算次数，包括理论浮点峰值和实测浮点峰值。理论浮点峰值是该计算机理论上每秒可以完成的浮点计算次数，主要由CPU的主频决定。理论浮点峰值＝CPU主频×CPU核数×CPU每周期执行浮点运算的次数。本文将为您介绍如何利用HPL测试实测浮点峰值。

测试HPC平台的HPL性能
--------------------

添加HPL.dat内容

.. code::

   mkdir ~/hpl
   cd ~/hpl
   touch HPL.dat
   touch hpl.lurm

HPL.dat内容如下所示

.. code::

   HPLinpack benchmark input file
   Innovative Computing Laboratory, University of Tennessee
   HPL.out      output file name (if any)
   6            device out (6=stdout,7=stderr,file)
   1            # of problems sizes (N)
   150272       Ns
   1            # of NBs
   256          NBs
   0            PMAP process mapping (0=Row-,1=Column-major)
   1            # of process grids (P x Q)
   4            Ps
   10           Qs
   16.0         threshold
   3            # of panel fact
   0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
   2            # of recursive stopping criterium
   2 4          NBMINs (>= 1)
   1            # of panels in recursion
   2            NDIVs
   3            # of recursive panel fact.
   0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
   1            # of broadcast
   0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
   1            # of lookahead depth
   0            DEPTHs (>=0)
   2            SWAP (0=bin-exch,1=long,2=mix)
   64           swapping threshold
   0            L1 in (0=transposed,1=no-transposed) form
   0            U  in (0=transposed,1=no-transposed) form
   1            Equilibration (0=no,1=yes)
   8            memory alignment in double (> 0)

hpl.slurm脚本内容如下

.. code::

   #!/bin/bash
   #SBATCH --job-name=hpl
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load hpl/2.3-intel-2021.4.0
   mpirun -np $SLURM_NTASKS xhpl

提交上述脚本

.. code::

   sbatch hpl.slurm

运行结果如下所示

.. code::

        ================================================================================
        T/V                N    NB     P     Q               Time                 Gflops
        --------------------------------------------------------------------------------
        WR11R2L4      234240   384     8     8            2420.15             3.5404e+03
        HPL_pdgesv() start time Tue Jan 11 21:19:09 2022

        HPL_pdgesv() end time   Tue Jan 11 21:59:35 2022

        --VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-
        Max aggregated wall time rfact . . . :               6.75
        + Max aggregated wall time pfact . . :               2.18
        + Max aggregated wall time mxswp . . :               1.34
        Max aggregated wall time update  . . :            2412.51
        + Max aggregated wall time laswp . . :             198.74
        Max aggregated wall time up tr sv  . :               0.68
        --------------------------------------------------------------------------------
        ||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   1.13372865e-03 ...... PASSED
        ================================================================================

测试ARM平台的HPL性能
--------------------

### 测试1个ARM节点的HPL性能

准备如下的HPL算例输入文件 ``HPL.dat`` 作为计算输入文件。


.. code::

    HPLinpack benchmark input file
    Innovative Computing Laboratory, University of Tennessee
    HPL.out      output file name (if any) 
    6            device out (6=stdout,7=stderr,file)
    1            # of problems sizes (N)
    163840         Ns
    1            # of NBs
    192           NBs
    0            PMAP process mapping (0=Row-,1=Column-major)
    1            # of process grids (P x Q)
    4            Ps
    8            Qs
    16.0         threshold
    1            # of panel fact
    0            PFACTs (0=left, 1=Crout, 2=Right)
    1            # of recursive stopping criterium
    2            NBMINs (>= 1)
    1            # of panels in recursion
    2            NDIVs
    1            # of recursive panel fact.
    0            RFACTs (0=left, 1=Crout, 2=Right)
    1            # of broadcast
    0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
    1            # of lookahead depth
    0            DEPTHs (>=0)
    2            SWAP (0=bin-exch,1=long,2=mix)
    64           swapping threshold
    0            L1 in (0=transposed,1=no-transposed) form
    0            U  in (0=transposed,1=no-transposed) form
    1            Equilibration (0=no,1=yes)
    8            memory alignment in double (> 0)
    ##### This line (no. 32) is ignored (it serves as a separator). ######
    0                               Number of additional problem sizes for PTRANS
    1200 10000 30000                values of N
    0                               number of additional blocking sizes for PTRANS
    40 9 8 13 13 20 16 32 64        values of NB

准备内容如下的作业脚本 ``hpl.slurm`` ，这个作业脚在一个节点上启动128个MPI进程、每个MPI进程启动一个OpenMP线程运行HPL基准测试。

.. code:: bash

    #!/bin/bash
    
    #SBATCH --job-name=hpl
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 128 
    #SBATCH --ntasks-per-node=128
    
    ulimit -l unlimited
    ulimit -s unlimited
    
    module load hpl/2.3-gcc-9.3.0-openblas-openmpi

    export NUM_OMP_THREADS=1

    cp `which xhpl` ./
    srun --mpi=pmi2 xhpl

使用Intel预编译程序测试HPL性能
------------------------------

.. tip:: Intel HPL使用时建议在每一个NUMA Socket启动一个MPI进程，然后再由MPI进程启动与Socket核心数匹配的计算线程。由于Intel HPL不使用OpenMP库，因此无法通过OMP环境变量控制计算线程数。

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

运行多节点Intel HPL性能测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO: 广超

参考资料
--------

- Running the Intel Distribution for LINPACK Benchmark https://www.intel.com/content/www/us/en/develop/documentation/onemkl-linux-developer-guide/top/intel-oneapi-math-kernel-library-benchmarks/intel-distribution-for-linpack-benchmark-1/run-the-intel-distribution-for-linpack-benchmark.html
- OSU Benchmarks gromacs官方网站 http://mvapich.cse.ohio-state.edu/benchmarks/
- DOWNLOAD, COMPILE AND RUN THE OSU BENCHMARK on AWS https://www.hpcworkshops.com/07-efa/04-complie-run-osu.html
- HOW DO I TUNE MY HPL.DAT FILE? https://www.advancedclustering.com/act_kb/tune-hpl-dat-file/
