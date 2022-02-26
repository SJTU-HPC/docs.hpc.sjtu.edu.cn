HPCG
====

简介
----

高性能共轭梯度（HPCG）基准程序旨在创建一个新的HPC系统排名指标。HPCG可以作为高性能LINPACK（HPL）基准的补充，该基准目前用于排名500强计算系统。HPL的计算和数据访问模式仍然代表着一些重要的可扩展应用，但并非必须。

HPCG程序可有效评估下述基本操作的性能：

``Sparse matrix-vector multiplication``

``Vector updates``

``Global dot products``

``Local symmetric Gauss-Seidel smoother``

``Sparse triangular solve (as part of the Gauss-Seidel smoother)``

``Driven by multigrid preconditioned conjugate gradient algorithm that exercises the key kernels on a nested set of coarse grids``

数据获取
--------

HPCG基准程序不需要直接读取数据，仅需改变两个重要的参数 ``problem_size`` 和 ``run_time_in_seconds`` ，这两个参数均可在运行脚本中指定。
problem_size应该设置的足够大，可使应用运行至少占用存储空间的25%；run_time_in_seconds官方规定应设置为1800s，但是
为了更快的得到结果，可以设置的小一些

如何在思源一号上导入HPCG环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   module load oneapi/2021.4.0
   cp -r $MKLROOT/benchmarks/hpcg ./
   cd hpcg

HPCG运行脚本(使用2个计算节点，每个节点使用2个进程，一个进程使用32个线程)

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=2nodes_hpcg
   #SBATCH --partition=64c512g
   #SBATCH -n 4
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=32
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi/2021.4.0
   export OMP_NUM_THREADS=32
   export KMP_AFFINITY=granularity=fine,compact,1,0
   export problem_size=192
   export run_time_in_seconds=60
   
   mpiexec.hydra -genvall -n 4 -ppn 2 bin/xhpcg_avx  -n$problem_size -t$run_time_in_seconds

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

