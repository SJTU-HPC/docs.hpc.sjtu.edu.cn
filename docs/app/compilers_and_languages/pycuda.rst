.. _pycuda:

PyCUDA
==========

简介
----

Nvidia的CUDA为我们提供了一种便捷的方式来直接操纵GPU进行并行编程，但是基于C语言的CUDA实现较为复杂，开发周期较长。而python 作为一门广泛使用的语言，具有简单易学、语法简单、开发迅速等优点。PyCUDA可以通过将python和C语言结合的方式进行CUDA编程。





PyCUDA安装以及使用说明
-----------------------------

思源一号上的PyCUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录PyCUDAtest并进入该目录：

.. code::

    mkdir PyCUDAtest
    cd PyCUDAtest

2. 申请计算资源并通过conda安装PyCUDA

.. code::

    srun -p 64c512g -n 10 --pty /bin/bash
    module load miniconda3
    conda create -n PyCUDAtest
    source activate PyCUDAtest
    conda install -c conda-forge pycuda

3. 在该目录下创建如下测试文件test.py：

.. code::

    import pycuda.driver as drv
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import numpy

    # 定义核函数
    mod = SourceModule(
        """
        __global__ void add_vectors(float *a, float *b, float *c, int n)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n)
            {
                c[idx] = a[idx] + b[idx];
            }
        }
    """
    )

    # 定义向量大小
    n = 10000

    # 生成随机向量数据
    a = numpy.random.randn(n).astype(numpy.float32)
    b = numpy.random.randn(n).astype(numpy.float32)

    # 分配输出内存空间
    c = numpy.zeros_like(a)

    # 将输入输出数据复制到 GPU
    a_gpu = drv.mem_alloc(a.nbytes)
    b_gpu = drv.mem_alloc(b.nbytes)
    c_gpu = drv.mem_alloc(c.nbytes)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)

    # 定义块和网格大小
    blocksize = 256
    gridsize = (n + blocksize - 1) // blocksize

    # 执行核函数
    add_vectors = mod.get_function("add_vectors")
    add_vectors(
        a_gpu, b_gpu, c_gpu, numpy.int32(n), block=(blocksize, 1, 1), grid=(gridsize, 1)
    )

    # 将结果从 GPU 复制回 CPU
    drv.memcpy_dtoh(c, c_gpu)

    # 检查计算结果是否正确
    assert numpy.allclose(c, a + b), "result not correct"

    # 输出结果
    print("a:", a)
    print("b:", b)
    print("c:", c)




4. 在该目录下创建如下作业提交脚本pycudatest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=pycudatest        # 作业名
    #SBATCH --partition=a100             # a100 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited

    module load miniconda3
    source activate PyCUDAtest
    module load cuda/11.5.0

    python3 test.py

5. 使用如下命令提交作业：

.. code::

  sbatch pycudatest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

    a: [ 0.32799047 -0.03553623 -1.6576846  ... -0.44243634 -1.1451671
     -1.1334891 ]
    b: [-0.46226323  0.76997334 -0.06620226 ...  0.6974032   2.1895697
      1.2849816 ]
    c: [-0.13427275  0.7344371  -1.7238868  ...  0.25496686  1.0444026
      0.15149248]





pi2.0上的PyCUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 申请计算资源并通过conda安装PyCUDA

.. code::

    srun -p cpu -N 1 --ntasks-per-node 40    --pty /bin/bash
    module load miniconda3
    conda create -n PyCUDAtest
    source activate PyCUDAtest
    conda install -c conda-forge pycuda



3. 此步骤和上文完全相同；

4. 在该目录下创建如下作业提交脚本mpi4pytest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=pycudatest        # 作业名
    #SBATCH --partition=dgx2             # dgx2 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited

    module load miniconda3
    source activate PyCUDAtest
    module load cuda/11.6.2-gcc-8.3.0

    python3 test.py


5. 此步骤和上文完全相同；
6. 此步骤和上文完全相同；









参考资料
-----------


-  `PyCUDA github <https://github.com/inducer/pycuda>`__

