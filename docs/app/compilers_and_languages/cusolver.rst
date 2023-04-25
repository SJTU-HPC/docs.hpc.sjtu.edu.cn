.. _cusolver:

cuSOLVER
==========

简介
----

The NVIDIA cuSOLVER library provides a collection of dense and sparse direct linear solvers and Eigen solvers which deliver significant acceleration for Computer Vision, CFD, Computational Chemistry, and Linear Optimization applications. The cuSOLVER library is included in both the NVIDIA HPC SDK and the CUDA Toolkit.



cuSOLVER使用说明
-----------------------------

思源一号上的cuSOLVER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录cuSOLVERtest并进入该目录：

.. code::

    mkdir cuSOLVERtest
    cd cuSOLVERtest

2. 在该目录下创建如下测试文件cuSOLVERtest.cu：

.. code::

    //利用LU分解法求解线性方程组Ax=b

    #include <cstdio>
    #include <cstdlib>
    #include <vector>

    #include <cuda_runtime.h>
    #include <cusolverDn.h>

    template <typename T>
    void print_matrix(const int &m, const int &n, const T *A, const int &lda);

    template <>
    void print_matrix(const int &m, const int &n, const float *A, const int &lda)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::printf("%0.2f ", A[j * lda + i]);
            }
            std::printf("\n");
        }
    }

    template <>
    void print_matrix(const int &m, const int &n, const double *A, const int &lda)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::printf("%0.2f ", A[j * lda + i]);
            }
            std::printf("\n");
        }
    }

    template <>
    void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
            }
            std::printf("\n");
        }
    }

    template <>
    void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
            }
            std::printf("\n");
        }
    }

    int main(int argc, char *argv[])
    {
        cusolverDnHandle_t cusolverH = NULL;
        cudaStream_t stream = NULL;

        const int m = 3;
        const int lda = m;
        const int ldb = m;

        /*
         *       | 1 2 3  |
         *   A = | 4 5 6  |
         *       | 7 8 10 |
         *
         * without pivoting: A = L*U
         *       | 1 0 0 |      | 1  2  3 |
         *   L = | 4 1 0 |, U = | 0 -3 -6 |
         *       | 7 2 1 |      | 0  0  1 |
         *
         * with pivoting: P*A = L*U
         *       | 0 0 1 |
         *   P = | 1 0 0 |
         *       | 0 1 0 |
         *
         *       | 1       0     0 |      | 7  8       10     |
         *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
         *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
         */

        const std::vector<double> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
        const std::vector<double> B = {1.0, 2.0, 3.0};
        std::vector<double> X(m, 0);
        std::vector<double> LU(lda * m, 0);
        std::vector<int> Ipiv(m, 0);
        int info = 0;

        double *d_A = nullptr; /* device copy of A */
        double *d_B = nullptr; /* device copy of B */
        int *d_Ipiv = nullptr; /* pivoting sequence */
        int *d_info = nullptr; /* error info */

        int lwork = 0;            /* size of workspace */
        double *d_work = nullptr; /* device workspace for getrf */

        const int pivot_on = 0;

        if (pivot_on)
        {
            printf("pivot is on : compute P*A = L*U \n");
        }
        else
        {
            printf("pivot is off: compute A = L*U (not numerically stable)\n");
        }

        printf("A = (matlab base-1)\n");
        print_matrix(m, m, A.data(), lda);
        printf("=====\n");

        printf("B = (matlab base-1)\n");
        print_matrix(m, 1, B.data(), ldb);
        printf("=====\n");

        /* step 1: create cusolver handle, bind a stream */
        (cusolverDnCreate(&cusolverH));

        (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        (cusolverDnSetStream(cusolverH, stream));

        /* step 2: copy A to device */
        (cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
        (cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
        (cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
        (cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

        (cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
        (cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));

        /* step 3: query working space of getrf */
        (cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));

        (cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

        /* step 4: LU factorization */
        if (pivot_on)
        {
            (cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
        }
        else
        {
            (cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info));
        }

        if (pivot_on)
        {
            (cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(), cudaMemcpyDeviceToHost, stream));
        }
        (cudaMemcpyAsync(LU.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));
        (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        (cudaStreamSynchronize(stream));

        if (0 > info)
        {
            printf("%d-th parameter is wrong \n", -info);
            exit(1);
        }
        if (pivot_on)
        {
            printf("pivoting sequence, matlab base-1\n");
            for (int j = 0; j < m; j++)
            {
                printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
            }
        }
        printf("L and U = (matlab base-1)\n");
        print_matrix(m, m, LU.data(), lda);
        printf("=====\n");

        /*
         * step 5: solve A*X = B
         *       | 1 |       | -0.3333 |
         *   B = | 2 |,  X = |  0.6667 |
         *       | 3 |       |  0      |
         *
         */
        if (pivot_on)
        {
            (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info));
        }
        else
        {
            (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, NULL, d_B, ldb, d_info));
        }

        (cudaMemcpyAsync(X.data(), d_B, sizeof(double) * X.size(), cudaMemcpyDeviceToHost, stream));
        (cudaStreamSynchronize(stream));

        printf("X = (matlab base-1)\n");
        print_matrix(m, 1, X.data(), ldb);
        printf("=====\n");

        /* free resources */
        (cudaFree(d_A));
        (cudaFree(d_B));
        (cudaFree(d_Ipiv));
        (cudaFree(d_info));
        (cudaFree(d_work));

        (cusolverDnDestroy(cusolverH));

        (cudaStreamDestroy(stream));

        (cudaDeviceReset());

        return EXIT_SUCCESS;
    }




3. 在该目录下创建如下作业提交脚本cuSOLVERtest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cuSOLVERtest        # 作业名
    #SBATCH --partition=a100             # a100 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load cuda/11.5.0
    module load gcc/11.2.0

    nvcc cuSOLVERtest.cu -o cuSOLVERtest -lcusolver
    ./cuSOLVERtest

4. 使用如下命令提交作业：

.. code::

  sbatch cuSOLVERtest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    pivot is off: compute A = L*U (not numerically stable)
    A = (matlab base-1)
    1.00 2.00 3.00
    4.00 5.00 6.00
    7.00 8.00 10.00
    =====
    B = (matlab base-1)
    1.00
    2.00
    3.00
    =====
    L and U = (matlab base-1)
    1.00 2.00 3.00
    4.00 -3.00 -6.00
    7.00 2.00 1.00
    =====
    X = (matlab base-1)
    -0.33
    0.67
    0.00
    =====



pi2.0上的cuSOLVER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本cuSOLVERtest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cuSOLVERStest        # 作业名
    #SBATCH --partition=dgx2             # dgx2 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load cuda/11.6.2-gcc-8.3.0
    module load gcc/8.3.0

    nvcc cuSOLVERtest.cu -o cuSOLVERtest -lcusolver
    ./cuSOLVERtest

4. 使用如下命令提交作业：

.. code::

  sbatch cuSOLVERtest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    pivot is off: compute A = L*U (not numerically stable)
    A = (matlab base-1)
    1.00 2.00 3.00
    4.00 5.00 6.00
    7.00 8.00 10.00
    =====
    B = (matlab base-1)
    1.00
    2.00
    3.00
    =====
    L and U = (matlab base-1)
    1.00 2.00 3.00
    4.00 -3.00 -6.00
    7.00 2.00 1.00
    =====
    X = (matlab base-1)
    -0.33
    0.67
    0.00
    =====






参考资料
-----------

-  `cuSOLVER 官网 <https://docs.nvidia.com/cuda/cusolver/index.html>`__
-  `cuSOLVER github <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER>`__

