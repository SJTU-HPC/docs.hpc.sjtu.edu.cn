.. _armadillo:

Armadillo
============

简介
----

Armadillo是目前使用比较广的C++矩阵运算库之一，是在C++下使用Matlab方式操作矩阵很好的选择，许多matlab的矩阵操作函数都可以找到对应，这对习惯了matlab的人来说实在是非常方便。他们之间的接口调用方式非常相似。



Armadillo使用说明
-----------------------------

思源一号上的Armadillo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录armadillotest并进入该目录：

.. code::

    mkdir armadillotest
    cd armadillotest

2. 在该目录下创建如下测试文件armadillotest.cpp：

.. code::

    // 对矩阵A进行PLU分解，其中P为置换矩阵,PA=LU

    #include <random>
    #include <algorithm>
    #include <stdlib.h>
    #include <iostream>
    #include <ctime>
    #include <armadillo>

    using namespace arma;

    int main()
    {
        int N = 5;
        arma::mat A = zeros<mat>(N, N);
        for (size_t i = 0; i < N; i++)
        {
            A(i, i) = i + 1.1;
            if (i != N - 1)
            {
                A(i, i + 1) = i + 4.4;
                A(i + 1, i) = i + 3.3;
            }
            if (i != N - 1 && i != N - 2)
            {
                A(i + 2, i) = i + 3.3;
            }
        }

        // 定义一个permutation matrix对象p
        arma::mat P;
        arma::mat L;
        arma::mat U;

        // 使用Armadillo的lu函数进行LU分解
        struct timespec start1, end1;
        double duration1;
        clock_gettime(CLOCK_REALTIME, &start1);

        arma::lu(L, U, P, A);

        clock_gettime(CLOCK_REALTIME, &end1);
        duration1 = (end1.tv_sec - start1.tv_sec) + (end1.tv_nsec - start1.tv_nsec) / 1000000000.0;
        printf("The time is %lf microseconds\n", duration1 * 1000);

        std::cout << "A:\n"
                  << A << std::endl;
        std::cout << "P:\n"
                  << P << std::endl;
        std::cout << "L:\n"
                  << L << std::endl;
        std::cout << "U:\n"
                  << U << std::endl;
        std::cout << "L*U:\n"
                  << L * U << std::endl;
        std::cout << "P*A:\n"
                  << P * A << std::endl;

        return 0;
    }




3. 在该目录下创建如下作业提交脚本armadillotest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=armadillotest
  #SBATCH --partition=64c512g
  #SBATCH --ntasks-per-node=1
  #SBATCH -n 1
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load armadillo/10.5.0-gcc-11.2.0-openblas
  module load gcc/11.2.0

  g++  armadillotest.cpp  -larmadillo -o armadillotest

  ./armadillotest

4. 使用如下命令提交作业：

.. code::

  sbatch armadillotest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    The time is 0.094627 microseconds
    A:
       1.1000   4.4000        0        0        0
       3.3000   2.1000   5.4000        0        0
       3.3000   4.3000   3.1000   6.4000        0
            0   4.3000   5.3000   4.1000   7.4000
            0        0   5.3000   6.3000   5.1000

    P:
            0   1.0000        0        0        0
            0        0        0   1.0000        0
       1.0000        0        0        0        0
            0        0   1.0000        0        0
            0        0        0        0   1.0000

    L:
       1.0000        0        0        0        0
            0   1.0000        0        0        0
       0.3333   0.8605   1.0000        0        0
       1.0000   0.5116   0.7879   1.0000        0
            0        0  -0.8333   0.4745   1.0000

    U:
       3.3000   2.1000   5.4000        0        0
            0   4.3000   5.3000   4.1000   7.4000
            0        0  -6.3605  -3.5279  -6.3674
            0        0        0   7.0821   1.2311
            0        0        0        0  -0.7899

    L*U:
       3.3000   2.1000   5.4000        0        0
            0   4.3000   5.3000   4.1000   7.4000
       1.1000   4.4000        0        0        0
       3.3000   4.3000   3.1000   6.4000        0
            0        0   5.3000   6.3000   5.1000

    P*A:
       3.3000   2.1000   5.4000        0        0
            0   4.3000   5.3000   4.1000   7.4000
       1.1000   4.4000        0        0        0
       3.3000   4.3000   3.1000   6.4000        0
            0        0   5.3000   6.3000   5.1000







参考资料
-----------

-  `Armadillo 官网 <https://arma.sourceforge.net/download.html>`__
-  `Armadillo 知乎 <https://zhuanlan.zhihu.com/p/442893337>`__

