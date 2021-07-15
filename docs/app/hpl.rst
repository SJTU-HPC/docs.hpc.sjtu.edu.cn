HPL
===

简介
----

HPL是什么。

+------------------+-------------+
| 模块名           | 平台        |
+==================+=============+
| hpl@2.3          | |cpu| |arm| |
+------------------+-------------+

HPL（The High-Performance Linpack Benchmark）是测试高性能计算集群系统浮点性能的基准。HPL通过对高性能计算集群采用高斯消元法求解一元N次稠密线性代数方程组的测试，评价高性能计算集群的浮点计算能力。

浮点计算峰值是指计算机每秒可以完成的浮点计算次数，包括理论浮点峰值和实测浮点峰值。理论浮点峰值是该计算机理论上每秒可以完成的浮点计算次数，主要由CPU的主频决定。理论浮点峰值＝CPU主频×CPU核数×CPU每周期执行浮点运算的次数。本文将为您介绍如何利用HPL测试实测浮点峰值。


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

提交到 ``arm128c256g`` 队列，随机测试1个ARM节点的HPL性能。
提交时添加 ``--nodelist`` 参数还能测试指定两个节点间的通信延迟。

.. code:: bash

    $ sbatch -p arm128c256g hpl.slurm

最后在 ``JOBID.out`` 文件中查看延迟测试值。这是100G Infiniband的参考结果。

.. code::

    # OSU MPI Latency Test v5.6.3
    # Size          Latency (us)
    0                       1.27
    1                       1.25
    2                       1.24
    4                       1.25
    8                       1.25
    16                      1.26
    32                      1.29
    64                      1.43
    128                     1.47
    256                     1.95
    512                     2.23
    1024                    2.77
    2048                    3.61
    4096                    4.86
    8192                    7.20
    16384                   9.93
    32768                  15.40
    65536                  26.64
    131072                 49.34
    262144                 27.79
    524288                 49.03
    1048576                91.58
    2097152               176.82
    4194304               346.91

使用 ``osu_mbw_mr`` 测量点对点通信带宽
--------------------------------------

准备内容如下的作业脚本 ``osu_bw.slurm`` ，这个作业脚本在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信带宽。

.. code:: bash

    #!/bin/bash
    
    #SBATCH --job-name=osu_bw
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=1
    #SBATCH --exclusive
    
    ulimit -l unlimited
    ulimit -s unlimited
    
    module load osu-micro-benchmarks/5.6.3-gcc-9.3.0-openmpi
    
    srun --mpi=pmi2 osu_mbw_mr

提交到 ``arm128c256g`` 队列，随机测试两个ARM节点间的MPI通信带宽。

.. code:: bash

    $ sbatch -p arm128c256g osu_bw.slurm

提交时添加 ``--nodelist`` 参数还能测试指定两个节点间的通信延迟。

.. code:: bash

    $ sbatch -p arm128c256g --nodelist=kp[020-021] osu_bw.slurm

最后在 ``JOBID.out`` 文件中查看带宽测试结果。这是100G Infiniband MPI带宽的参考结果。

.. code::

    # OSU MPI Multiple Bandwidth / Message Rate Test v5.6.3
    # [ pairs: 1 ] [ window size: 64 ]
    # Size                  MB/s        Messages/s
    1                       4.24        4235302.84
    2                       8.82        4409629.80
    4                      17.55        4387775.11
    8                      34.67        4333726.75
    16                     67.82        4238584.63
    32                    129.61        4050327.86
    64                    262.59        4102908.64
    128                   499.14        3899519.14
    256                   811.93        3171585.76
    512                  1529.29        2986902.43
    1024                 2068.14        2019668.41
    2048                 2700.72        1318710.75
    4096                 3399.47         829948.38
    8192                 3878.01         473390.04
    16384               11338.92         692072.80
    32768               11810.79         360436.61
    65536               12074.32         184239.48
    131072              12190.81          93008.50
    262144              12266.13          46791.59
    524288              12305.57          23471.02
    1048576             12324.26          11753.33
    2097152             12335.56           5882.05
    4194304             12340.24           2942.14

参考资料
--------

- OSU Benchmarks gromacs官方网站 http://mvapich.cse.ohio-state.edu/benchmarks/
- DOWNLOAD, COMPILE AND RUN THE OSU BENCHMARK on AWS https://www.hpcworkshops.com/07-efa/04-complie-run-osu.html
- HOW DO I TUNE MY HPL.DAT FILE? https://www.advancedclustering.com/act_kb/tune-hpl-dat-file/
