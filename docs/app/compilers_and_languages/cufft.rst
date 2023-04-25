.. _cufft:

cuFFT
==========

简介
----

The cuFFT is a CUDA Fast Fourier Transform library consisting of two components: cuFFT and cuFFTW. The cuFFT library provides high performance on NVIDIA GPUs, and the cuFFTW library is a porting tool to use FFTW on NVIDIA GPUs.



cuFFT使用说明
-----------------------------

思源一号上的cuFFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录cuFFTtest并进入该目录：

.. code::

    mkdir cuFFTtest
    cd cuFFTtest

2. 在该目录下创建如下测试文件cuFFTtest.cu：

.. code::
    
    //  In this example a one-dimensional complex-to-complex transform is applied to the input data.
    //Afterwards an inverse transform is performed on the computed frequency domain representation.

    #include <complex>
    #include <iostream>
    #include <random>
    #include <vector>

    #include <cuda_runtime.h>
    #include <cufftXt.h>

    // #include "cufft_utils.h"

    // CUDA API error checking
    #define CUDA_RT_CALL(call)                                                     \
        {                                                                          \
            auto status = static_cast<cudaError_t>(call);                          \
            if (status != cudaSuccess)                                             \
                fprintf(stderr,                                                    \
                        "ERROR: CUDA RT call \"%s\" in line %d of file %s failed " \
                        "with "                                                    \
                        "%s (%d).\n",                                              \
                        #call,                                                     \
                        __LINE__,                                                  \
                        __FILE__,                                                  \
                        cudaGetErrorString(status),                                \
                        status);                                                   \
        }

    // cufft API error chekcing
    #define CUFFT_CALL(call)                                                     \
        {                                                                        \
            auto status = static_cast<cufftResult>(call);                        \
            if (status != CUFFT_SUCCESS)                                         \
                fprintf(stderr,                                                  \
                        "ERROR: CUFFT call \"%s\" in line %d of file %s failed " \
                        "with "                                                  \
                        "code (%d).\n",                                          \
                        #call,                                                   \
                        __LINE__,                                                \
                        __FILE__,                                                \
                        status);                                                 \
        }

    int main(int argc, char *argv[])
    {
        cufftHandle plan;
        cudaStream_t stream = NULL;

        int n = 8;
        int batch_size = 2;
        int fft_size = batch_size * n;

        using scalar_type = float;
        using data_type = std::complex<scalar_type>;

        std::vector<data_type> data(fft_size);

        for (int i = 0; i < fft_size; i++)
        {
            data[i] = data_type(i, -i);
        }

        std::printf("Input array:\n");
        for (auto &i : data)
        {
            std::printf("%f + %fj\n", i.real(), i.imag());
        }
        std::printf("=====\n");

        cufftComplex *d_data = nullptr;

        CUFFT_CALL(cufftCreate(&plan));
        CUFFT_CALL(cufftPlan1d(&plan, data.size(), CUFFT_C2C, batch_size));

        CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUFFT_CALL(cufftSetStream(plan, stream));

        // Create device data arrays
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(data_type) * data.size()));
        CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(), cudaMemcpyHostToDevice, stream));

        /*
         * Note:
         *  Identical pointers to data and output arrays implies in-place transformation
         */
        CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

        CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(), cudaMemcpyDeviceToHost, stream));

        CUDA_RT_CALL(cudaStreamSynchronize(stream));

        std::printf("Output array:\n");
        for (auto &i : data)
        {
            std::printf("%f + %fj\n", i.real(), i.imag());
        }
        std::printf("=====\n");

        /* free resources */
        CUDA_RT_CALL(cudaFree(d_data))

        CUFFT_CALL(cufftDestroy(plan));

        CUDA_RT_CALL(cudaStreamDestroy(stream));

        CUDA_RT_CALL(cudaDeviceReset());

        return EXIT_SUCCESS;
    }



3. 在该目录下创建如下作业提交脚本cuFFTtest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cuFFTtest        # 作业名
    #SBATCH --partition=a100             # a100 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load cuda/11.5.0
    module load gcc/11.2.0

    nvcc cuFFTtest.cu -o cuFFTtest -lcufft
    ./cuFFTtest

4. 使用如下命令提交作业：

.. code::

  sbatch cuFFTtest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    Input array:
    0.000000 + 0.000000j
    1.000000 + -1.000000j
    2.000000 + -2.000000j
    3.000000 + -3.000000j
    4.000000 + -4.000000j
    5.000000 + -5.000000j
    6.000000 + -6.000000j
    7.000000 + -7.000000j
    8.000000 + -8.000000j
    9.000000 + -9.000000j
    10.000000 + -10.000000j
    11.000000 + -11.000000j
    12.000000 + -12.000000j
    13.000000 + -13.000000j
    14.000000 + -14.000000j
    15.000000 + -15.000000j
    =====
    Output array:
    0.000004 + 0.000000j
    16.000015 + -16.000004j
    32.000004 + -32.000004j
    48.000004 + -48.000004j
    64.000000 + -64.000000j
    80.000000 + -80.000008j
    96.000000 + -96.000015j
    112.000015 + -112.000000j
    128.000000 + -128.000000j
    144.000000 + -144.000000j
    160.000000 + -160.000000j
    176.000000 + -176.000000j
    192.000000 + -192.000000j
    208.000000 + -208.000000j
    224.000000 + -223.999985j
    239.999985 + -240.000000j
    =====



pi2.0上的cuFFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本cuFFTtest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cuFFTtest        # 作业名
    #SBATCH --partition=dgx2             # dgx2 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load cuda/11.6.2-gcc-8.3.0
    module load gcc/8.3.0

    nvcc cuFFTtest.cu -o cuFFTtest -lcufft
    ./cuFFTtest

4. 使用如下命令提交作业：

.. code::

  sbatch cuFFTtest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    Input array:
    0.000000 + 0.000000j
    1.000000 + -1.000000j
    2.000000 + -2.000000j
    3.000000 + -3.000000j
    4.000000 + -4.000000j
    5.000000 + -5.000000j
    6.000000 + -6.000000j
    7.000000 + -7.000000j
    8.000000 + -8.000000j
    9.000000 + -9.000000j
    10.000000 + -10.000000j
    11.000000 + -11.000000j
    12.000000 + -12.000000j
    13.000000 + -13.000000j
    14.000000 + -14.000000j
    15.000000 + -15.000000j
    =====
    Output array:
    0.000004 + 0.000000j
    16.000015 + -16.000004j
    32.000004 + -32.000004j
    48.000004 + -48.000004j
    64.000000 + -64.000000j
    80.000000 + -80.000008j
    96.000000 + -96.000015j
    112.000015 + -112.000000j
    128.000000 + -128.000000j
    144.000000 + -144.000000j
    160.000000 + -160.000000j
    176.000000 + -176.000000j
    192.000000 + -192.000000j
    208.000000 + -208.000000j
    224.000000 + -223.999985j
    239.999985 + -240.000000j
    =====






参考资料
-----------

-  `cuFFT 官网 <https://docs.nvidia.com/cuda/cufft/index.html>`__
-  `cuFFT github <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFT>`__

