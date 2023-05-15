.. _mpi4py:

Mpi4py
==========

简介
----

Mpi4py 是一个构建在 MPI 之上的 Python 库，它使得 Python 的数据结构可以方便的在多进程中传递。Mpi4py 是一个很强大的库，它实现了很多 MPI 标准中的接口，包括点对点通信，集合通信、阻塞／非阻塞通信、组间通信等，基本上能用到的 MPI 接口都有相应的实现。不仅是任何可以被 pickle 的 Python 对象，Mpi4py 对具有单段缓冲区接口的 Python 对象如 numpy 数组及内置的 bytes/string/array 等也有很好的支持并且传递效率很高。同时它还提供了 SWIG 和 F2PY 的接口能够将 C/C++ 或者 Fortran 程序在封装成 Python 后仍然能够使用 Mpi4py 的对象和接口来进行并行处理。





Mpi4py安装以及使用说明
-----------------------------

思源一号上的Mpi4py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录Mpi4pytest并进入该目录：

.. code::

    mkdir Mpi4pytest
    cd Mpi4pytest

2. 申请计算资源并通过conda安装mpi4py

.. code::

    srun -p 64c512g -n 10 --pty /bin/bash
    module load miniconda3
    conda create -n mpi4py425
    source activate mpi4py425
    conda install -c conda-forge mpi4py

3. 在该目录下创建如下测试文件test.py：

.. code::

    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD  # 获取通信子
    rank = comm.Get_rank()  # 获取当前进程的rank
    size = comm.Get_size()  # 获取进程的总数

    sendbuf = None  # 定义发送缓冲区

    if rank == 0:
        # 在根进程中初始化数组
        sendbuf = np.arange(size * 2, dtype='i')
        print("Sendbuf:", sendbuf)

    # 定义接收缓冲区
    recvbuf = np.empty(2, dtype='i')

    # 将数据从根进程分发到其他进程中
    comm.Scatter(sendbuf, recvbuf, root=0)
    print("Process %d received: %s" % (rank, recvbuf))


4. 在该目录下创建如下作业提交脚本mpi4pytest.slurm:

.. code::

    #!/bin/bash

    #BATCH --job-name=mpi4pytest      # 作业名
    #SBATCH --partition=64c512g      # 64c512g队列
    #SBATCH --ntasks-per-node=10     # 每节点核数
    #SBATCH -n 10                     # 作业核心数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited

    module load miniconda3
    source activate mpi4py425

    mpirun -np 10  python3 test.py

5. 使用如下命令提交作业：

.. code::

  sbatch mpi4pytest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

    Sendbuf: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    Process 0 received: [0 1]
    Process 8 received: [16 17]
    Process 4 received: [8 9]
    Process 6 received: [12 13]
    Process 7 received: [14 15]
    Process 5 received: [10 11]
    Process 9 received: [18 19]
    Process 2 received: [4 5]
    Process 1 received: [2 3]
    Process 3 received: [6 7]



pi2.0上的Mpi4py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 申请计算资源并通过conda安装mpi4py

.. code::

    srun -p cpu -N 1 --ntasks-per-node 40    --pty /bin/bash
    module load miniconda3
    conda create -n mpi4py425
    source activate mpi4py425
    conda install -c conda-forge mpi4py



3. 此步骤和上文完全相同；
4. 在该目录下创建如下作业提交脚本mpi4pytest.slurm:

.. code::

    #!/bin/bash

    #BATCH --job-name=mpi4pytest      # 作业名
    #SBATCH --partition=small        # small队列
    #SBATCH --ntasks-per-node=10     # 每节点核数
    #SBATCH -n 10                     # 作业核心数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited

    module load miniconda3
    source activate mpi4py425

    mpirun -np 10  python3 test.py


5. 此步骤和上文完全相同；
6. 此步骤和上文完全相同；









参考资料
-----------

-  `Mpi4py github <https://github.com/mpi4py/mpi4py>`__
-  `Mpi4py 知乎 <https://zhuanlan.zhihu.com/p/157804393>`__

