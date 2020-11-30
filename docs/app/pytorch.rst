.. _pytorch:

PyTorch
=======

简介
----

PyTorch 是一个 Python 优先的深度学习框架，也是使用 GPU 和 CPU
优化的深度学习张量库，能够在强大的 GPU
加速基础上实现张量和动态神经网络。同时，PyTorch
主要为开发者提供两种高层面的功能：

1. 使用强大的 GPU 加速的 Tensor 计算（类似 numpy）；
2. 构建 autograd 系统的深度神经网络。

通常，人们使用 PyTorch 的原因通常有二：

1. 作为 numpy 的替代，以便使用强大的 GPU；
2. 将其作为一个能提供最大的灵活性和速度的深度学习研究平台

π 集群上的 PyTorch
----------------------

π 集群上可用 miniconda 自行安装 PyTorch，也可用已预置的 NVIDIA 提供的 NGC
镜像 ``pytorch-1.6.0``\ （性能更好）

使用 miniconda 安装 PyTorch
---------------------------------

创建名为 ``pytorch-env`` 的虚拟环境，激活虚拟环境，然后安装 pytorch

.. code:: bash

   $ module load miniconda3
   $ conda create -n pytorch-env
   $ source activate pytorch-env
   $ conda install pytorch torchvision -c pytorch

提交 PyTorch 作业
----------------------

示例：在 DGX-2 上使用 pytorch。作业使用单节点，分配 2 块 GPU，GPU:CPU
配比 1:6。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2

   module load miniconda3
   source activate pytorch-env

   python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'

使用以下指令提交作业

.. code:: bash

   $ sbatch slurm.test

使用 π 集群提供的 PyTorch
--------------------------------

π 集群中已经预置了 `NVIDIA GPU CLOUD <https://ngc.nvidia.com/>`__
提供的优化镜像，通过调用该镜像即可运行 PyTorch
作业，无需单独安装，目前版本为 ``pytorch/1.6.0``\ 。

查看 Pi 上已编译的软件模块:

.. code:: bash

   module av pytorch

调用该模块:

.. code:: bash

   module load pytorch/1.6.0

以下 slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu
核心。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2

   module load pytorch/1.6.0

   python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

-  `PyTorch官网 <https://pytorch.org/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
