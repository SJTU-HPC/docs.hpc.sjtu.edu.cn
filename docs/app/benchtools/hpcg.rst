HPCG
====

简介
----

高性能共轭梯度（HPCG）基准程序旨在创建一个新的HPC系统排名指标。HPCG可以作为高性能LINPACK（HPL）基准的补充，该基准目前用于排名500强计算系统。HPL的计算和数据访问模式仍然代表着一些重要的可扩展应用，但并非必须。HPCG的设计目的是使计算和数据访问模式更紧密地匹配一些重要的应用程序。

如何在思源一号上导入HPCG环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   module load oneapi/2021.4.0
   cp -r $MKLROOT/benchmarks/hpcg ./

HPCG基准程序有两个重要的参数, ``Problem size`` 和 ``Run time`` 。
Problem size应该设置的足够大，可使应用运行至少占用存储空间的25%；Run time官方规定应设置为1800s，但是为了更快的得到结果，可以设置的小一些。

使用如下脚本运行HPCG(使用2个计算节点，单节点使用2个进程，一个进程使用32个线程)

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
   
   mpiexec.hydra -genvall -n 4 -ppn 2 bin/xhpcg_avx  -n$problem_size -t$run_time_in_seconds

运行结果如下所示：


