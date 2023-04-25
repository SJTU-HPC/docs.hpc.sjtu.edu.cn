.. _eigen:

Eigen
==========

简介
----

Eigen库是一个开源的矩阵运算库，其利用C++模板编程的思想，构造所有矩阵通过传递模板参数形式完成。由于模板类不支持库链接方式编译，而且模板类要求全部写在头文件中，从而导致导致Eigen库只能通过开源的方式供大家使用，并且只需要包含Eigen头文件就能直接使用。



Eigen使用说明
-----------------------------

思源一号上的Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录eigentest并进入该目录：

.. code::
        
    mkdir eigentest
    cd eigentest

2. 在该目录下创建如下测试文件eigentest.cpp：

.. code::
        
    // 利用Eigen提供的SparseLU求解器求解稀疏线性方程组Ax=b
    #include <vector>
    #include <Eigen/Sparse>
    #include <Eigen/SparseLU>
    #include <chrono>
    #include <random>
    #include <algorithm>
    #include <stdlib.h>
    #include <iostream>
    #include <ctime>

    using namespace std;
    using namespace Eigen;
    // using namespace chrono;

    typedef SparseMatrix<double> SpMat;
    typedef Triplet<double> T;

    int main()
    {
        int N = 200;

        Eigen::SparseMatrix<double> A(N, N); // 创建稀疏矩阵
        for (size_t i = 0; i < N; i++)
        {
            A.insert(i, i) = i + 1.1;
            if (i != N - 1)
            {
                A.insert(i, i + 1) = i + 4.4;
                A.insert(i + 1, i) = i + 3.3;
            }
            if (i != N - 1 && i != N - 2)
            {
                A.insert(i + 2, i) = i + 3.3;
            }
        }

        // A.makeCompressed(); // 将稀疏矩阵转换为压缩列格式
        // cout << "A = " << A << endl;

        Eigen::VectorXd b(N); // 创建 5 维向量
        for (size_t i = 0; i < N; i++)
        {
            b(i) = i + 1;
        }

        struct timespec start1, end1;
        double duration1;
        clock_gettime(CLOCK_REALTIME, &start1);

        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        Eigen::VectorXd x = solver.solve(b);

        clock_gettime(CLOCK_REALTIME, &end1);
        duration1 = (end1.tv_sec - start1.tv_sec) + (end1.tv_nsec - start1.tv_nsec) / 1000000000.0;
        printf("Time taken by solving is %lf microseconds\n", duration1 * 1000);

        for (size_t i = 0; i < 10; i++)
        {
            printf("x[%d] = %f\n", i, x[i]);
        }
        cout << "===================================================" << endl;

        return 0;
    }


3. 在该目录下创建如下作业提交脚本eigentest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=eigentest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load eigen/3.4.0-gcc-11.2.0
  module load gcc/11.2.0

  g++  eigentest.cpp -o eigentest

  ./eigentest

4. 使用如下命令提交作业：

.. code::

  sbatch eigentest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    Time taken by solving is 2.991600 microseconds
    x[0] = 5.866133
    x[1] = -1.239260
    x[2] = -2.732554
    x[3] = -0.399766
    x[4] = 3.439243
    x[5] = 0.531062
    x[6] = -2.109301
    x[7] = -0.724832
    x[8] = 2.550861
    x[9] = 1.037857
    ===================================================



pi2.0上的Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本eigentest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=eigentest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load eigen/3.4.0-gcc-11.2.0
  module load gcc/11.2.0

  g++  eigentest.cpp -o eigentest

  ./eigentest

4. 使用如下命令提交作业：

.. code::

  sbatch eigentest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    Time taken by solving is 3.391600 microseconds
    x[0] = 5.866133
    x[1] = -1.239260
    x[2] = -2.732554
    x[3] = -0.399766
    x[4] = 3.439243
    x[5] = 0.531062
    x[6] = -2.109301
    x[7] = -0.724832
    x[8] = 2.550861
    x[9] = 1.037857
    ===================================================



  



参考资料
-----------

-  `Eigen 官网 <https://eigen.tuxfamily.org/index.php?title=Main_Page>`__
-  `Eigen 知乎 <https://zhuanlan.zhihu.com/p/462494086>`__

