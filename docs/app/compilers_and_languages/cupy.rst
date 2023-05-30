.. _cupy:

Cupy
==========

简介
----

CuPy 是一个借助 CUDA GPU 库在英伟达 GPU 上实现 Numpy 数组的库。基于 Numpy 数组的实现，GPU 自身具有的多个 CUDA 核心可以促成更好的并行加速。
CuPy 接口是 Numpy 的一个镜像，并且在大多情况下，它可以直接替换 Numpy 使用。只要用兼容的 CuPy 代码替换 Numpy 代码，用户就可以实现 GPU 加速。
CuPy 支持 Numpy 的大多数数组运算，包括索引、广播、数组数学以及各种矩阵变换。





Cupy安装以及使用说明
-----------------------------

思源一号上的Cupy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录Cupytest并进入该目录：

.. code::

    mkdir Cupytest
    cd Cupytest

2. 申请计算资源并通过conda安装cupy

.. code::

    srun -p 64c512g -n 10 --pty /bin/bash
    module load miniconda3
    conda create -n cupytest
    source activate cupytest
    conda install -c conda-forge cupy

3. 在该目录下创建如下测试文件test.py：

.. code::

    import numpy as np
    import cupy as cp
    import time

    start_time = time.time()
    x0 = np.ones((100,1000,1000))
    x1= 5*x0
    X2= x1*x1
    end_time = time.time()
    print('numpy执行用时：',end_time - start_time)

    start_time1 = time.time()
    x3 = cp.ones((100,1000,1000))
    x4= 5*x3
    X5= x4*x4
    end_time1 = time.time()
    print('cupy执行用时：',end_time1 - start_time1)



4. 在该目录下创建如下作业提交脚本cupytest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cupytest          # 作业名
    #SBATCH --partition=a100             # a100 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load miniconda3
    source activate cupytest

    python3 test.py

5. 使用如下命令提交作业：

.. code::

  sbatch cupytest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

    numpy执行用时： 0.5541410446166992
    cupy执行用时： 0.3665752410888672



pi2.0上的Cupy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 申请计算资源并通过conda安装cupy

.. code::

    srun -p cpu -N 1 --ntasks-per-node 40    --pty /bin/bash
    module load miniconda3
    conda create -n cupytest
    source activate cupytest
    conda install -c conda-forge cupy



3. 此步骤和上文完全相同；
4. 在该目录下创建如下作业提交脚本cupytest.slurm:

.. code::

    #!/bin/bash

    #SBATCH --job-name=cupytest          # 作业名
    #SBATCH --partition=dgx2             # dgx2 队列
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1            # 1:1 的 GPU:CPU 配比
    #SBATCH --gres=gpu:1                 # 1 块 GPU
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load miniconda3
    source activate cupytest

    python3 test.py


5. 此步骤和上文完全相同；
6. 此步骤和上文完全相同；









参考资料
-----------

-  `Cupy github <https://github.com/cupy/cupy>`__
-  `Cupy 知乎 <https://zhuanlan.zhihu.com/p/594460098>`__