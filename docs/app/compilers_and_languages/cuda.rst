CUDA
====

思源一号平台及AI平台上均部署有CUDA套件。

+----------+------------------------------------+
| 版本     | 加载方式                           |
+==========+====================================+        
| 10.1.243 | module load cuda/10.1.243 思源一号 |
+----------+------------------------------------+
| 11.3.1   | module load cuda/11.3.1   思源一号 |
+----------+------------------------------------+
| 11.4.0   | module load cuda/11.4.0   思源一号 |
+----------+------------------------------------+
| 11.5.0   | module load cuda/11.5.0   思源一号 |
+----------+------------------------------------+
| 11.8.0   | module load cuda/11.8.0   思源一号 |
+----------+------------------------------------+
| 8.0.61   | module load cuda/8.0.61-gcc-4.8.5  |
+----------+------------------------------------+
| 9.0.176  | module load cuda/9.0.176-gcc-4.8.5 |
+----------+------------------------------------+
| 9.2.88   | module load cuda/9.2.88-gcc-4.8.5  |
+----------+------------------------------------+
| 10.0.130 | module load cuda/10.0.130-gcc-4.8.5|
+----------+------------------------------------+
| 10.1.243 | module load cuda/10.1.243-gcc-4.8.5|
+----------+------------------------------------+

本文档向您展示如何使用CUDA，包含程序示例，编译，作业脚本示例。


程序示例 CuBLAS
------------------

编辑 ``cublas.cu`` 文件，内容如下：

.. code:: cuda

   //Example. Application Using C and CUBLAS: 1-based indexing
   //-----------------------------------------------------------
   #include <stdio.h>
   #include <stdlib.h>
   #include <math.h>
   #include <cuda_runtime.h>
   #include "cublas_v2.h"
   #define M 6
   #define N 5
   #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

   static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
       cublasSscal (handle, n-q+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
       cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
   }

   int main (void){
       cudaError_t cudaStat;    
       cublasStatus_t stat;
       cublasHandle_t handle;
       int i, j;
       float* devPtrA;
       float* a = 0;
       a = (float *)malloc (M * N * sizeof (*a));
       if (!a) {
           printf ("host memory allocation failed");
           return EXIT_FAILURE;
       }
       for (j = 1; j <= N; j++) {
           for (i = 1; i <= M; i++) {
               a[IDX2F(i,j,M)] = (float)((i-1) * M + j);
           }
       }
       cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
       if (cudaStat != cudaSuccess) {
           printf ("device memory allocation failed");
           return EXIT_FAILURE;
       }
       stat = cublasCreate(&handle);
       if (stat != CUBLAS_STATUS_SUCCESS) {
           printf ("CUBLAS initialization failed\n");
           return EXIT_FAILURE;
       }
       stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
       if (stat != CUBLAS_STATUS_SUCCESS) {
           printf ("data download failed");
           cudaFree (devPtrA);
           cublasDestroy(handle);
           return EXIT_FAILURE;
       }
       modify (handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
       stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
       if (stat != CUBLAS_STATUS_SUCCESS) {
           printf ("data upload failed");
           cudaFree (devPtrA);
           cublasDestroy(handle);        
           return EXIT_FAILURE;
       }    
       cudaFree (devPtrA);
       cublasDestroy(handle);
       for (j = 1; j <= N; j++) {
           for (i = 1; i <= M; i++) {
               printf ("%7.0f", a[IDX2F(i,j,M)]);
           }
           printf ("\n");
       }
       free(a);
       return EXIT_SUCCESS;
   }

将以上程序保存为 `cublas.cu`。

使用cuda进行编译，编译时链接cublas动态库。

在思源平台上的编译命令如下：

.. code:: bash

   $ module load cuda/11.3.1
   $ nvcc cublas.cu -o cublas -lcublas

在AI平台上的编译命令如下：

.. code:: bash

   $ module load cuda/10.0.130-gcc-4.8.5
   $ nvcc cublas.cu -o cublas -lcublas


a100队列作业脚本示例
--------------------------

这是一个名为 ``a100.slurm`` 的 **单机单卡**
作业脚本，该脚本向a100队列申请1块GPU，并在作业完成时通知。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=cuda_test
   #SBATCH --partition=a100
   #SBATCH --gres=gpu:1
   #SBATCH -N 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=cublas.out
   #SBATCH --error=cublas.err

   module load cuda/11.3.1

   ./cublas

用以下方式提交作业：

.. code:: bash

   $ sbatch a100.slurm

DGX2队列作业脚本示例
-----------------------

这是一个名为 ``dgx.slurm`` 的 **单机单卡**
作业脚本，该脚本向dgx2队列申请1块GPU，并在作业完成时通知。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=dgx2_test
   #SBATCH --partition=dgx2
   #SBATCH --gres=gpu:1
   #SBATCH -N 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=cublas.out
   #SBATCH --error=cublas.err

   module load cuda/10.0.130-gcc-4.8.5

   ./cublas

用以下方式提交作业：

.. code:: bash

   $ sbatch dgx.slurm

预期结果
----------------------

.. code:: bash

   $ cat cublas.out
          1      7     13     19     25     31
          2      8     14     20     26     32
          3   1728    180    252    324    396
          4    160     16     22     28     34
          5    176     17     23     29     35
