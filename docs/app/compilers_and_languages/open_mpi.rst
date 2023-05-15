.. _open_mpi:

OpenMPI
==========

简介
----

OpenMPI是一个免费的、开源的MPI实现，兼容MPI-1和MPI-2标准。OpenMPI由开源社区开发维护，支持大多数类型的HPC平台，并具有很高的性能。



OpenMPI使用说明
-----------------------------

思源一号上的OpenMPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录openmpitest并进入该目录：

.. code::
        
    mkdir openmpitest
    cd openmpitest

2. 在该目录下创建如下测试文件openmpitest.c：

.. code::

    #include "stdio.h"
    #include "mpi.h"
    #include "omp.h"
    #include "math.h"

    #define NUM_THREADS 8
    long int n = 10000000;

    int main(int argc, char *argv[])
    {
        int my_rank, numprocs;
        long int i, my_n, my_first_i, my_last_i;
        double my_pi = 0.0, pi, h, x;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        h = 1.0 / n;
        my_n = n / numprocs;
        my_first_i = my_rank * my_n;
        my_last_i = my_first_i + my_n;

        omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for reduction(+ : my_pi) private(x, i)
        for (i = my_first_i; i < my_last_i; i++)
        {
            x = (i + 0.5) * h;
            my_pi = my_pi + 4.0 / (1.0 + x * x);
        }

        MPI_Reduce(&my_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (my_rank == 0)
        {
            printf("Approximation of pi:%15.13f\n", pi * h);
        }

        MPI_Finalize();

        return 0;
    }



3. 在该目录下创建如下作业提交脚本openmpitest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=openmpitest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 4                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load gcc
  module load openmpi/4.1.1-gcc-8.3.1

  mpicc openmpitest.c -o openmpitest -fopenmp

  mpirun -np 4 ./openmpitest

4. 使用如下命令提交作业：

.. code::

  sbatch openmpitest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

  Approximation of pi:3.1415926535898

pi2.0上的OpenMPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本openmpitest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=openmpitest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 4                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load gcc
  module load openmpi/3.1.5-gcc-9.2.0

  mpicc openmpitest.c -o openmpitest -fopenmp

  mpirun -np 4 ./openmpitest

4. 使用如下命令提交作业：

.. code::

  sbatch openmpitest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

   Approximation of pi:3.1415926535898


  



参考资料
---------

-  `OpenMPI 入门教程 <https://zhuanlan.zhihu.com/p/399150417>`__






