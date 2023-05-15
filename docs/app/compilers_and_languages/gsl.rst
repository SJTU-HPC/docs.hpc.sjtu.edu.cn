.. _GSL:

GSL
==========

简介
----

GSL(GNU Scientific Library)是专门为应用数学和科学技术领域的数值计算提供支持的软件库。GSL使用C语言编写，同时也为其他语言做了相应的封装。GSL在GNU通用公共许可下是免费的。该函数库提供了广泛的数学算法的实现函数，包括随机数生成器，特殊函数和最小二乘拟合等等。目前该函数库提供有超过1000个函数，这些函数包含的范围有：复数计算、多项式求根、特殊函数、向量和矩阵运算、排列、组合、排序、线性代数、特征值和特征向量、快速傅里叶变换(FFT)、数值积分、随机数生成、随机数分布、统计、蒙特卡洛积分、模拟退火、常微分方程组、插值、数值微分、方程求根、最小二乘拟合、小波变换等。




GSL使用说明
-----------------------------

思源一号上的GSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 创建gsltest目录并进入该目录：

.. code::
        
  mkdir gsltest
  cd gsltest

2. 在该目录下编写如下mygsl.c文件：

.. code::
        
    #include <stdio.h>
    #include <gsl/gsl_linalg.h>

    int main()
    {
        double a_data[] = {1.0, 0.6, 0.0, 0.0, 1.5, 1.0, 0.0, 1.0, 1.0};
        double inva[9];
        int s, i, j;

        gsl_matrix_view m = gsl_matrix_view_array(a_data, 3, 3);
        gsl_matrix_view inv = gsl_matrix_view_array(inva, 3, 3);
        gsl_permutation *p = gsl_permutation_alloc(3);

        printf("The matrix is\n");
        for (i = 0; i < 3; ++i)
        {
            for (j = 0; j < 3; ++j)
            {
                printf(j == 2 ? "%6.3f\n" : "%6.3f ", gsl_matrix_get(&m.matrix, i, j));
            }
        }

        gsl_linalg_LU_decomp(&m.matrix, p, &s);
        gsl_linalg_LU_invert(&m.matrix, p, &inv.matrix);

        printf("The inverse is\n");
        for (i = 0; i < 3; ++i)
        {
            for (j = 0; j < 3; ++j)
            {
                printf(j == 2 ? "%6.3f\n" : "%6.3f ", gsl_matrix_get(&inv.matrix, i, j));
            }
        }

        gsl_permutation_free(p);

        return 0;
    }


3. 在该目录下编写如下gsltest.slurm脚本：

.. code::

  #!/bin/bash

  #SBATCH --job-name=gsltest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load gcc/9.3.0
  module load gsl/2.7-gcc-9.3.0

  gcc mygsl.c -o mygsl -lgsl -lgslcblas -lm

  ./mygsl


4. 使用如下命令提交作业：

.. code::

  sbatch gsltest.slurm


5. 作业完成后在.out文件中可看到如下结果：

.. code::

   The matrix is
   1.000  0.600  0.000
   0.000  1.500  1.000
   0.000  1.000  1.000
   The inverse is
   1.000 -1.200  1.200
   0.000  2.000 -2.000
   0.000 -2.000  3.000


pi2.0上的GSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；

2. 此步骤和上文完全相同；

3. 在该目录下编写如下gsltest.slurm脚本：

.. code::

  #!/bin/bash

  #SBATCH --job-name=gsltest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load gcc/9.2.0
  module load gsl/2.5-gcc-9.2.0

  gcc mygsl.c -o mygsl -lgsl -lgslcblas -lm

  ./mygsl


4. 使用如下命令提交作业：

.. code::

  sbatch gsltest.slurm


5. 作业完成后在.out文件中可看到如下结果：

.. code::

   The matrix is
   1.000  0.600  0.000
   0.000  1.500  1.000
   0.000  1.000  1.000
   The inverse is
   1.000 -1.200  1.200
   0.000  2.000 -2.000
   0.000 -2.000  3.000


参考资料
----------


-  `GSL官方文档 <https://www.gnu.org/software/gsl/doc/html/index.html>`__



