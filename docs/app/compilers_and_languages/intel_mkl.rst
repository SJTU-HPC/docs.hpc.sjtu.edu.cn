.. _intel_mkl:

Intel mkl
==========

简介
----

Intel mkl是一套经过高度优化和广泛线程化的数学例程，专为需要极致性能的科学、工程及金融等领域的应用而设计。核心数学函数包括 BLAS、LAPACK、ScaLAPACK1、稀疏矩阵解算器、快速傅立叶转换、矢量数学及其它函数。
它可以为当前及下一代英特尔处理器提供性能优化，包括更出色地与 Microsoft Visual Studio、Eclipse和XCode相集成。Intel mkl支持完全集成英特尔兼容性OpenMPI运行时库，以实现更出色的 Windows/Linux跨平台兼容性。





Intel mkl使用说明
-----------------------------

思源一号上的Intel mkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录intelmkltest并进入该目录：

.. code::
        
    mkdir intelmkltest
    cd intelmkltest

2. 在该目录下创建如下测试文件intelmkltest.c：

.. code::
        
  #include <stdio.h>
  #include <stdlib.h>
  #include "mkl.h"

  #define min(x,y) (((x) < (y)) ? (x) : (y))

  int main()
  {
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");

    m = 2000, k = 200, n = 1000;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    printf ("\n Computations completed.\n\n");

    printf (" Top left corner of matrix A: \n");
    for (i=0; i<min(m,6); i++) {
        for (j=0; j<min(k,6); j++) {
            printf ("%12.0f", A[j+i*k]);
        }
        printf ("\n");
    }

    printf ("\n Top left corner of matrix B: \n");
    for (i=0; i<min(k,6); i++) {
        for (j=0; j<min(n,6); j++) {
            printf ("%12.0f", B[j+i*n]);
        }
        printf ("\n");
    }

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(m,6); i++) {
        for (j=0; j<min(n,6); j++) {
            printf ("%12.5G", C[j+i*n]);
        }
        printf ("\n");
    }

    printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf (" Example completed. \n\n");
    return 0;
  }

3. 在该目录下创建如下作业提交脚本intelmkltest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=intelmkltest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited


  module load intel-oneapi-compilers/2021.4.0
  module load intel-mkl/2020.4.304
  

  icc  intelmkltest.c -o intelmkltest -qmkl

  ./intelmkltest

4. 使用如下命令提交作业：

.. code::

  sbatch intelmkltest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

  This example computes real matrix C=alpha*A*B+beta*C using 
  Intel(R) MKL function dgemm, where A, B, and  C are matrices and 
  alpha and beta are double precision scalars

  Initializing data for matrix multiplication C=A*B for matrix 
  A(2000x200) and matrix B(200x1000)

  Allocating memory for matrices aligned on 64-byte boundary for better 
  performance 

  Intializing matrix data 

  Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface 


  Computations completed.

  Top left corner of matrix A: 
           1           2           3           4           5           6
         201         202         203         204         205         206
         401         402         403         404         405         406
         601         602         603         604         605         606
         801         802         803         804         805         806
        1001        1002        1003        1004        1005        1006

  Top left corner of matrix B: 
          -1          -2          -3          -4          -5          -6
       -1001       -1002       -1003       -1004       -1005       -1006
       -2001       -2002       -2003       -2004       -2005       -2006
       -3001       -3002       -3003       -3004       -3005       -3006
       -4001       -4002       -4003       -4004       -4005       -4006
       -5001       -5002       -5003       -5004       -5005       -5006

  Top left corner of matrix C: 
  -2.6666E+09 -2.6666E+09 -2.6667E+09 -2.6667E+09 -2.6667E+09 -2.6667E+09
  -6.6467E+09 -6.6467E+09 -6.6468E+09 -6.6468E+09 -6.6469E+09  -6.647E+09
  -1.0627E+10 -1.0627E+10 -1.0627E+10 -1.0627E+10 -1.0627E+10 -1.0627E+10
  -1.4607E+10 -1.4607E+10 -1.4607E+10 -1.4607E+10 -1.4607E+10 -1.4607E+10
  -1.8587E+10 -1.8587E+10 -1.8587E+10 -1.8587E+10 -1.8588E+10 -1.8588E+10
  -2.2567E+10 -2.2567E+10 -2.2567E+10 -2.2567E+10 -2.2568E+10 -2.2568E+10

  Deallocating memory 

  Example completed. 


pi2.0上的Intel mkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本intelmkltest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=intelmkltest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load intel-oneapi-compilers/2021.4.0
  module load intel-mkl/2019.3.199

  icc  intelmkltest.c -o intelmkltest -qmkl

  ./intelmkltest

4. 使用如下命令提交作业：

.. code::

  sbatch intelmkltest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

  This example computes real matrix C=alpha*A*B+beta*C using 
  Intel(R) MKL function dgemm, where A, B, and  C are matrices and 
  alpha and beta are double precision scalars

  Initializing data for matrix multiplication C=A*B for matrix 
  A(2000x200) and matrix B(200x1000)

  Allocating memory for matrices aligned on 64-byte boundary for better 
  performance 

  Intializing matrix data 

  Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface 


  Computations completed.

  Top left corner of matrix A: 
           1           2           3           4           5           6
         201         202         203         204         205         206
         401         402         403         404         405         406
         601         602         603         604         605         606
         801         802         803         804         805         806
        1001        1002        1003        1004        1005        1006

  Top left corner of matrix B: 
          -1          -2          -3          -4          -5          -6
       -1001       -1002       -1003       -1004       -1005       -1006
       -2001       -2002       -2003       -2004       -2005       -2006
       -3001       -3002       -3003       -3004       -3005       -3006
       -4001       -4002       -4003       -4004       -4005       -4006
       -5001       -5002       -5003       -5004       -5005       -5006

  Top left corner of matrix C: 
  -2.6666E+09 -2.6666E+09 -2.6667E+09 -2.6667E+09 -2.6667E+09 -2.6667E+09
  -6.6467E+09 -6.6467E+09 -6.6468E+09 -6.6468E+09 -6.6469E+09  -6.647E+09
  -1.0627E+10 -1.0627E+10 -1.0627E+10 -1.0627E+10 -1.0627E+10 -1.0627E+10
  -1.4607E+10 -1.4607E+10 -1.4607E+10 -1.4607E+10 -1.4607E+10 -1.4607E+10
  -1.8587E+10 -1.8587E+10 -1.8587E+10 -1.8587E+10 -1.8588E+10 -1.8588E+10
  -2.2567E+10 -2.2567E+10 -2.2567E+10 -2.2567E+10 -2.2568E+10 -2.2568E+10

  Deallocating memory 

  Example completed. 


  



参考资料
----------

-  `Intel mkl 官网教程 <https://software.intel.com/en-us/mkl-tutorial-c-overview>`__

