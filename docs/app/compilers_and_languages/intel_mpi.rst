.. _intel_mpi:

Intel-MPI
========================

英特尔® MPI 库是一个实现开源 MPICH 规范的多结构消息传递库。使用该库创建、维护和测试高级、复杂的应用程序，这些应用程序在基于英特尔® 处理器的高性能计算 (HPC) 集群上表现更好。

加载预安装的Intel组件
---------------------

可以用以下命令加载集群中已安装的Intel组件:

+---------------------+----------------------------------+--------------------------+
| 版本                | 加载方式                         | 组件说明                 |
+=====================+==================================+==========================+        
| intel-mpi-2021.4.0  | module load                      | Intel MPI库              |
|                     | intel-oneapi-mpi/2021.4.0        |                          |
|                     | module load                      |                          |
|                     | intel-oneapi-compilers/2021.4.0  |                          |      
+---------------------+----------------------------------+--------------------------+
| intel-mpi-2019.     | module load                      | Intel MPI库              |
| 4.243               | intel-mpi/2019.4.243             |                          |
+---------------------+----------------------------------+--------------------------+
| intel-mpi-2019.     | module load                      | Intel MPI库              |
| 6.154               | intel-mpi/2019.6.154             |                          |
+---------------------+----------------------------------+--------------------------+



在使用intel-mpi的时候，请尽量保持编译器版本与后缀中的编译器版本一致，如intel-mpi-2019.4.243/intel-19.0.4和intel-19.0.4
另外我们建议直接使用Intel全家桶

使用Intel+Intel-mpi编译应用
---------------------------

这里，我们演示如何使用系统中的Intel和Intel-mpi编译MPI代码，使用的MPI代码可以在\ ``/lustre/share/samples/MPI/mpihello.c``\ 中找到。

加载和编译：

.. code:: bash

   $ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
   $ mpiicc mpihello.c -o mpihello

提交Intel+Intel-mpi应用
-----------------------

准备一个 mpihello.c 程序

.. code:: bash

   #include "mpi.h"
   #include <stdio.h>
   #include <stdlib.h>

   int  main(int argc,char* argv[])
   {
        int myid,numprocs;   
        int namelen;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
        MPI_Get_processor_name(processor_name,&namelen);
        printf("Hello World! Process %d of %d on %s\n",myid,numprocs,processor_name);
        MPI_Finalize();

        return 0;
    }



准备一个名为job_impi.slurm的作业脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=mpihello
   #SBATCH --partition=small
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 2

   ulimit -s unlimited
   ulimit -l unlimited

   module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

   mpiicc mpihello.c -o mpihello

   mpirun -np 2 ./mpihello

若采用 intel 2018，脚本中 export I_MPI_FABRICS=shm:ofi
这行需改为 export I_MPI_FABRICS=shm:tmi

最后，将作业提交到SLURM

.. code:: bash

   $ sbatch job_impi.slurm

参考资料
--------

-  `intel-parallel-studio <https://software.intel.com/zh-cn/parallel-studio-xe/>`__
