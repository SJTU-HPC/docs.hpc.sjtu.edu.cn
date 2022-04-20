HPCG
====

简介
----

HPCG(High Performance Conjugate Gradient)基准测试程序作为超级计算机系统的性能评估指标之一，可有效评估超算系统在下述基本操作中的性能表现：

``Sparse matrix-vector multiplication``

``Vector updates``

``Global dot products``

``Local symmetric Gauss-Seidel smoother``

``Sparse triangular solve (as part of the Gauss-Seidel smoother)``

``Driven by multigrid preconditioned conjugate gradient algorithm that exercises the key kernels on a nested set of coarse grids``

导入HPCG环境
------------

+--------+-------+----------+---------------------------+
| 版本   | 平台  | 构建方式 | 模块名                    |
+========+=======+==========+===========================+
| 3.1    | |cpu| | spack    | oneapi/2021.4.0  思源一号 |
+--------+-------+----------+---------------------------+
| 3.1    | |cpu| | spack    | oneapi/2021.4.0  π2.0     |
+--------+-------+----------+---------------------------+

.. code:: bash

   mkdir ~/HPCG && cd ~/HPCG
   module load oneapi/2021.4.0
   cp -r $MKLROOT/benchmarks/hpcg ./
   cd hpcg

HPCG运行的重要参数
------------------

 ``problem_size`` 和 ``run_time_in_seconds`` 在 ``hpcg/bin/hpcg.dat`` 中的默认值为 ``192``, ``60`` ，这两个参数均可在运行脚本中指定。

在思源一号上运行
----------------

HPCG运行脚本
(每个计算节点上共有两个CPU Socket，每个CPU Socket启动1个进程，每个计算节点启动2个进程)

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=2nodes_hpcg
   #SBATCH --partition=64c512g
   #SBATCH -n 4
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=32
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi/2021.4.0
   export OMP_NUM_THREADS=32
   export KMP_AFFINITY=granularity=fine,compact,1,0
   export problem_size=192
   export run_time_in_seconds=60
   
   mpiexec.hydra -genvall bin/xhpcg_avx  -n$problem_size -t$run_time_in_seconds

使用如下命令提交作业

.. code:: bash

   sbatch run_hyper.slurm

运行结束后，将产生如下文件，n192-4p-32t_V3.1_2022-02-09_16-43-22.txt，其中192代表问题规模，4代表使用的进程，32代表1个进程包含的线程数。

.. code:: bash

   Final Summary =
   Final Summary ::HPCG result is VALID with a GFLOP/s rating of=109.132
   Final Summary ::    HPCG 2.4 Rating (for historical value) is=109.691
   Final Summary ::Reference version of ComputeDotProduct used=Performance results are most likely suboptimal
   Final Summary ::Results are valid but execution time (sec) is=65.1121
   Final Summary ::     Official results execution time (sec) must be at least=1800

在π2.0上运行
-------------

HPCG运行脚本
(每个计算节点上共有两个CPU Socket，每个CPU Socket启动1个进程，每个计算节点启动2个进程)

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=2nodes_hpcg
   #SBATCH --partition=cpu
   #SBATCH -n 4
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=20
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load oneapi/2021.4.0
   export OMP_NUM_THREADS=20
   export KMP_AFFINITY=granularity=fine,compact,1,0
   export problem_size=192
   export run_time_in_seconds=60

   mpiexec.hydra -genvall bin/xhpcg_avx  -n$problem_size -t$run_time_in_seconds

使用如下命令提交作业

.. code:: bash

   sbatch run_hyper.slurm

运行结束后，将产生如下文件，n192-4p-20t_V3.1_2022-02-26_16-34-36.txt，其中192代表问题规模，4代表使用的进程，32>代表1个进程包含的线程数。

.. code:: bash

   Final Summary =
   Final Summary ::HPCG result is VALID with a GFLOP/s rating of=74.4941
   Final Summary ::    HPCG 2.4 Rating (for historical value) is=74.829
   Final Summary ::Reference version of ComputeDotProduct used=Performance results are most likely suboptimal
   Final Summary ::Results are valid but execution time (sec) is=62.6445
   Final Summary ::     Official results execution time (sec) must be at least=1800

运行结果
--------

思源一号
~~~~~~~~

+-------------------------------------------+
|  problem_size:192  run_time_in_seconds:60 |
+============+==============+===============+
| 核数       | 64           | 128           |
+------------+--------------+---------------+
| GFOLP/s    | 56.09485     | 112.07949     |
+------------+--------------+---------------+

π2.0
~~~~

+-------------------------------------------+
|  problem_size:192  run_time_in_seconds:60 |
+============+==============+===============+
| 核数       | 40           | 80            |
+------------+--------------+---------------+
| GFOLP/s    | 37.9614      | 74.4941       |
+------------+--------------+---------------+
