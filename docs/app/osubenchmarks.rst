OSU Benchmarks
==============

简介
----

OSU Benchmarks

超算平台通过模块提供了OSU Benchmarsk模块，可用 ``module load`` 指令加载。

+----------------------------------------------+-------------+
| 模块名                                       | 平台        |
+==============================================+=============+
| osu-micro-benchmarks/5.6.3-gcc-9.3.0-openmpi | |cpu| |arm| |
+----------------------------------------------+-------------+

使用 ``osu_latency`` 测量点对点通信延迟
---------------------------------------

准备内容如下的作业脚本 ``osu_latency.slurm`` ，这个作业脚本在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信延迟。

.. code:: bash

    #!/bin/bash
    
    #SBATCH --job-name=osu_latency
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=1
    #SBATCH --exclusive
    
    ulimit -l unlimited
    ulimit -s unlimited
    
    module load osu-micro-benchmarks/5.6.3-gcc-9.3.0-openmpi
    
    srun --mpi=pmi2 osu_latency

提交到 ``arm128c256g`` 队列，随机测试两个ARM节点间的MPI通信延迟。

.. code:: bash

    $ sbatch -p arm128c256g osu_latency.slurm

提交时添加 ``--nodelist`` 参数还能测试指定两个节点间的通信延迟。

.. code:: bash

    $ sbatch -p arm128c256g --nodelist=kp[020-021] osu_latency.slurm

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

- OSU Benchmarks http://mvapich.cse.ohio-state.edu/benchmarks/
- DOWNLOAD, COMPILE AND RUN THE OSU BENCHMARK on AWS https://www.hpcworkshops.com/07-efa/04-complie-run-osu.html
