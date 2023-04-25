.. _cusparse:

cuSPARSE
==========

简介
----

cuSPARSE 库为稀疏矩阵提供经 GPU 加速的基本线性代数子程序，与仅使用 CPU 的替代方案相比，这一程序的执行速度有显著的提升。此库提供了用于构建 GPU 加速型求解器的功能。在从事机器学习、计算流体力学、地震勘探及计算科学等应用的工程师和科学家群体中，cuSPARSE 得到了广泛采用。通过使用 cuSPARSE，应用将能自动从定期性能提升及全新 GPU 架构中受益。cuSPARSE 库包含在 NVIDIA HPC SDK 和 CUDA 工具包中。



cuSPARSE使用说明
-----------------------------

思源一号上的cuSPARSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录cuSPARSEtest并进入该目录：

.. code::

    mkdir cuSPARSEtest
    cd cuSPARSEtest

2. 在该目录下创建如下测试文件cuSPARSEtest.cu：

.. code::

    // 求解下三角稀疏线性方程组Ax=b

    #include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
    #include <cusparse.h>         // cusparseSpSV
    #include <stdio.h>            // printf
    #include <stdlib.h>           // EXIT_FAILURE

    #define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

    #define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

    int main(void)
    {
        // A = [1, 0, 0, 0
        //      0, 4, 0, 0
        //      5, 0, 6, 0
        //      0, 8, 0, 9]

        // b = [1
        //      8
        //      23
        //      52]

        // Host problem definition
        const int A_num_rows = 4;
        const int A_num_cols = 4;
        const int A_nnz = 6;
        int hA_csrOffsets[] = {0, 1, 2, 4, 6};
        int hA_columns[] = {0, 1, 0, 2, 1, 3};
        float hA_values[] = {1.0f, 4.0f, 5.0f, 6.0f, 8.0f, 9.0f};
        float hX[] = {1.0f, 8.0f, 23.0f, 52.0f};
        float hY[] = {0.0f, 0.0f, 0.0f, 0.0f};
        float hY_result[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float alpha = 1.0f;
        //--------------------------------------------------------------------------
        // Device memory management
        int *dA_csrOffsets, *dA_columns;
        float *dA_values, *dX, *dY;
        CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)))
        CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(float)))
        CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(float)))

        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice))
        //--------------------------------------------------------------------------
        // CUSPARSE APIs
        cusparseHandle_t handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void *dBuffer = NULL;
        size_t bufferSize = 0;
        cusparseSpSVDescr_t spsvDescr;
        CHECK_CUSPARSE(cusparseCreate(&handle))
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                         dA_csrOffsets, dA_columns, dA_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
        // Create dense vector X
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F))
        // Create dense vector y
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F))
        // Create opaque data structure, that holds analysis data between calls.
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr))
        // Specify Lower|Upper fill mode.
        cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode)))
        // Specify Unit|Non-Unit diagonal type.
        cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype)))
        // allocate an external buffer for analysis
        CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matA, vecX, vecY, CUDA_R_32F,
                                               CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                               &bufferSize))
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
        CHECK_CUSPARSE(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha, matA, vecX, vecY, CUDA_R_32F,
                                             CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer))
        // execute SpSV
        CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, vecX, vecY, CUDA_R_32F,
                                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr))

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseDestroySpMat(matA))
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
        CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
        CHECK_CUSPARSE(cusparseDestroy(handle))
        //--------------------------------------------------------------------------
        // device result check
        CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost))
        int correct = 1;
        for (int i = 0; i < A_num_rows; i++)
        {
            if (hY[i] != hY_result[i])
            {                // direct floating point comparison is not
                correct = 0; // reliable
                break;
            }
        }
        if (correct)
            printf("spsv_csr_example test PASSED\n");
        else
            printf("spsv_csr_example test FAILED: wrong result\n");
        for (size_t i = 0; i < A_num_rows; i++)
        {
            printf("x[%d] = %f\n", i, hY[i]);
        }
        //--------------------------------------------------------------------------
        // device memory deallocation
        CHECK_CUDA(cudaFree(dBuffer))
        CHECK_CUDA(cudaFree(dA_csrOffsets))
        CHECK_CUDA(cudaFree(dA_columns))
        CHECK_CUDA(cudaFree(dA_values))
        CHECK_CUDA(cudaFree(dX))
        CHECK_CUDA(cudaFree(dY))
        return EXIT_SUCCESS;
    }




3. 在该目录下创建如下作业提交脚本cuSPARSEtest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cuSPARSEtest        # 作业名
    #SBATCH --partition=a100             # a100 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load cuda/11.5.0
    module load gcc/11.2.0

    nvcc cuSPARSEtest.cu -o cuSPARSEtest -lcusparse
    ./cuSPARSEtest

4. 使用如下命令提交作业：

.. code::

  sbatch cuSPARSEtest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    spsv_csr_example test PASSED
    x[0] = 1.000000
    x[1] = 2.000000
    x[2] = 3.000000
    x[3] = 4.000000



pi2.0上的cuSPARSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本cuSPARSEtest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cuSPARSEtest        # 作业名
    #SBATCH --partition=dgx2             # dgx2 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load cuda/11.6.2-gcc-8.3.0
    module load gcc/8.3.0

    nvcc cuSPARSEtest.cu -o cuSPARSEtest -lcusparse
    ./cuSPARSEtest

4. 使用如下命令提交作业：

.. code::

  sbatch cuSPARSEtest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    spsv_csr_example test PASSED
    x[0] = 1.000000
    x[1] = 2.000000
    x[2] = 3.000000
    x[3] = 4.000000








参考资料
-----------

-  `cuSPARSE 官网 <https://docs.nvidia.com/cuda/cusparse/index.html>`__
-  `cuSPARSE github <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE>`__

