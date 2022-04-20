OSU Benchmarks
==============

简介
----

OSU Benchmarks

超算平台通过模块提供了OSU Benchmarsk模块，可用 ``module load`` 指令加载。

+-------+-------+----------+--------------------------------------------------------+
| 版本  | 平台  | 构建方式 | 模块名                                                 |
+=======+=======+==========+========================================================+
| 5.7.1 | |cpu| | spack    | osu-micro-benchmarks/5.7.1-gcc-11.2.0-openmpi 思源一号 |
+-------+-------+----------+--------------------------------------------------------+
| 5.7.1 | |cpu| | spack    | osu-micro-benchmarks/5.7.1-gcc-9.2.0-openmpi-4.0.5     |
+-------+-------+----------+--------------------------------------------------------+
| 5.6.3 | |cpu| | spack    | osu-micro-benchmarks/5.6.3-gcc-9.3.0-openmpi           |
+-------+-------+----------+--------------------------------------------------------+

集群上的OSU
------------

- `思源一号上的OSU`_

- `π2.0上的OSU`_

- `ARM上的OSU`_
  
.. _思源一号上的OSU:

思源一号上的OSU基准测试
------------------------

思源一号使用 osu_latency 测量点对点通信延迟
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本 ``osu_latency.slurm`` 在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信延迟。

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=osu_latency
    #SBATCH --partition=64c512g 
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=1
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.error
    
    module load openmpi
    module load osu-micro-benchmarks/5.7.1-gcc-11.2.0-openmpi
    
    mpirun osu_latency

思源一号上使用 osu_mbw_mr 测量点对点通信带宽
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本 ``osu_bw.slurm`` 在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信带宽。

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=osu_latency
    #SBATCH --partition=64c512g 
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=1
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.error
    
    module load openmpi
    module load osu-micro-benchmarks/5.7.1-gcc-11.2.0-openmpi
    
    mpirun  osu_mbw_mr

思源一号上使用 osu_bcast 测量广播延迟
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要注意广播延迟和参与测试的节点数量正相关，如果需要对比性能，要保证测试作业所用的节点数一致。

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=osu_latency
    #SBATCH --partition=64c512g 
    #SBATCH -n 4
    #SBATCH --ntasks-per-node=1
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.error
    
    module load openmpi
    module load osu-micro-benchmarks/5.7.1-gcc-11.2.0-openmpi
    
    mpirun osu_bcast

.. _π2.0上的OSU:

π2.0上的OSU基准测试
---------------------

π2.0上使用 osu_latency 测量点对点通信延迟
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本 ``osu_latency.slurm`` 在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信延迟。

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=osu
    #SBATCH --partition=cpu
    #SBATCH --exclusive 
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=1
    
    module load gcc/9.2.0
    module load openmpi/4.0.5-gcc-9.2.0
    module load osu-micro-benchmarks/5.7.1-gcc-9.2.0-openmpi-4.0.5
    
    mpirun osu_latency

π2.0上使用 osu_mbw_mr 测量点对点通信带宽
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本 ``osu_bw.slurm`` 在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信带宽。

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=osu
    #SBATCH --partition=cpu
    #SBATCH --exclusive 
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 2
    #SBATCH --ntasks-per-node=1
    
    module load gcc/9.2.0
    module load openmpi/4.0.5-gcc-9.2.0
    module load osu-micro-benchmarks/5.7.1-gcc-9.2.0-openmpi-4.0.5
    
    mpirun osu_mbw_mr

π2.0上使用 osu_bcast 测量广播延迟
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要注意广播延迟和参与测试的节点数量正相关，如果需要对比性能，要保证测试作业所用的节点数一致。

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=osu
    #SBATCH --partition=cpu
    #SBATCH --exclusive 
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 4
    #SBATCH --ntasks-per-node=1
    
    module load gcc/9.2.0
    module load openmpi/4.0.5-gcc-9.2.0
    module load osu-micro-benchmarks/5.7.1-gcc-9.2.0-openmpi-4.0.5

    mpirun osu_bcast

.. _ARM上的OSU:

ARM上的OSU基准测试
------------------

ARM上使用 osu_latency 测量点对点通信延迟
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本 ``osu_latency.slurm`` 在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信延迟。

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

ARM上使用 osu_mbw_mr 测量点对点通信带宽
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作业脚本 ``osu_bw.slurm`` 在两个节点上各启动一个MPI进程，测量两个MPI进程之间的通信带宽。

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

ARM上使用 osu_bcast 测量广播延迟
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要注意广播延迟和参与测试的节点数量正相关，如果需要对比性能，要保证测试作业所用的节点数一致。

.. code:: bash

    #!/bin/bash
    
    #SBATCH --job-name=osu_bw
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -n 4
    #SBATCH --ntasks-per-node=1
    #SBATCH --exclusive
    
    ulimit -l unlimited
    ulimit -s unlimited
    
    module load osu-micro-benchmarks/5.6.3-gcc-9.3.0-openmpi
    
    srun --mpi=pmi2 osu_bcast
    
测试结果
---------

OSU MPI Latency
~~~~~~~~~~~~~~~~~~

.. code:: bash
         
   # OSU MPI Latency Test
             思源一号 v5.7.1       π2.0 v5.7.1          ARM v5.6.3
   # Size       Latency (us)       Latency (us)       Latency (us)
   0                    0.79               1.55               1.27
   1                    0.79               1.34               1.25
   2                    0.79               1.29               1.24
   4                    0.79               1.25               1.25
   8                    0.78               1.24               1.25
   16                   0.79               1.59               1.26
   32                   0.82               1.59               1.29
   64                   0.91               1.50               1.43
   128                  0.95               1.51               1.47
   256                  1.24               1.56               1.95
   512                  1.26               1.63               2.23
   1024                 1.37               1.79               2.77
   2048                 2.08               2.11               3.61
   4096                 2.80               2.71               4.86
   8192                 3.85               3.98               7.20
   16384                5.73               9.11               9.93
   32768                7.62              12.15              15.40
   65536               10.62              23.43              26.64
   131072              15.85              32.53              49.34
   262144              21.32              44.96              27.79
   524288              39.55              65.61              49.03
   1048576             74.91             109.06              91.58
   2097152            145.99             199.42             176.82
   4194304            286.26             393.98             346.91

OSU MPI Multiple Bandwidth / Message Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

思源一号上的OSU MPI Multiple Bandwidth
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

   # OSU MPI Multiple Bandwidth / Message Rate Test v5.7.1
   # [ pairs: 1 ] [ window size: 64 ]
   # Size                  MB/s        Messages/s
   1                       6.82        6819901.32
   2                      13.70        6849644.94
   4                      27.43        6857747.81
   8                      54.70        6837453.43
   16                    109.97        6873169.62
   32                    218.58        6830520.90
   64                    402.61        6290822.77
   128                   773.02        6039234.26
   256                  1446.47        5650271.17
   512                  2646.67        5169286.04
   1024                 4411.59        4308188.99
   2048                 7656.33        3738444.41
   4096                10508.77        2565618.70
   8192                12463.12        1521377.49
   16384               13336.64         814004.02
   32768               13109.51         400070.54
   65536               13959.39         213003.44
   131072              14438.86         110159.77
   262144              14689.16          56034.71
   524288              14825.20          28276.83
   1048576             14887.78          14198.09
   2097152             14909.55           7109.43
   4194304             14910.01           3554.82

π2.0上的OSU MPI Multiple Bandwidth
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

   # OSU MPI Multiple Bandwidth / Message Rate Test v5.7.1
   # [ pairs: 1 ] [ window size: 64 ]
   # Size                  MB/s        Messages/s
   1                       1.45        1454746.80
   2                       2.91        1456550.86
   4                       7.50        1875912.49
   8                      14.89        1860936.30
   16                     29.09        1818298.04
   32                     65.96        2061307.14
   64                    130.36        2036819.98
   128                   270.56        2113729.90
   256                   586.45        2290813.80
   512                  1108.22        2164499.97
   1024                 1934.49        1889151.11
   2048                 3082.52        1505136.87
   4096                 4380.14        1069370.93
   8192                 6035.57         736763.84
   16384                4511.71         275372.74
   32768                6618.96         201994.78
   65536                9373.92         143034.69
   131072              11988.77          91467.04
   262144              12119.05          46230.50
   524288              12193.54          23257.32
   1048576             12226.86          11660.44
   2097152             12140.43           5789.01
   4194304             12108.15           2886.81

ARM上的OSU MPI Multiple Bandwidth
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

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

OSU MPI Broadcast Latency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
       
   # OSU MPI Broadcast Latency Test
                    思源一号                π2.0                  ARM
   # Size    Avg Latency(us)      Avg Latency(us)      Avg Latency(us)
   1                    0.45                 3.00                 2.35
   2                    0.44                 2.89                 2.43
   4                    0.45                 2.87                 2.38
   8                    0.44                 2.96                 2.37
   16                   0.45                 4.02                 2.42
   32                   0.47                 3.91                 2.42
   64                   0.87                 3.71                 2.59
   128                  0.66                 3.74                 2.66
   256                  1.01                 3.79                 3.11
   512                  1.11                 4.10                 3.36
   1024                 1.24                 4.04                 3.86
   2048                 3.35                 8.39                 4.20
   4096                 4.37                14.71                 5.25
   8192                 3.48                27.43                 7.15
   16384                5.48                53.14                11.37
   32768                9.43               107.36                19.00
   65536               15.92               205.86                34.18
   131072              28.89               415.82                65.70
   262144              55.15               849.62               132.38
   524288             107.83               385.97               122.73
   1048576            169.50               780.35               239.25 

参考资料
--------

- OSU Benchmarks http://mvapich.cse.ohio-state.edu/benchmarks/
- DOWNLOAD, COMPILE AND RUN THE OSU BENCHMARK on AWS https://www.hpcworkshops.com/07-efa/04-complie-run-osu.html
