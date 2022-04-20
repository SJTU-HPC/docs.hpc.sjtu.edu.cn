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


π 超算上的 PyTorch
----------------------

π 超算上可用 miniconda 自行安装 PyTorch，也可用已预置的 NVIDIA 提供的 NGC
镜像 ``pytorch-1.6.0``\ （性能更好）。

+----------+----------------+----------+---------------------------------------------------+
|版本      |平台            |构建方式  |名称                                               |
+==========+================+==========+===================================================+
| 1.6.0    |  |gpu|         | 容器     |/lustre/share/singularity/modules/pytorch/1.6.0.sif|
+----------+----------------+----------+---------------------------------------------------+
| 1.6.0    |  |gpu|         | 容器     |/dssg/share/imgs/pytorch/1.6.0.sif思源平台         |
+----------+----------------+----------+---------------------------------------------------+



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

示例：在 DGX-2 及 A100 上均可使用 pytorch。作业使用单节点，分配 2 块 GPU，GPU:CPU
配比 1:6。脚本名称可设为 slurm.test。

在DGX-2队列上提交pytorch作业：

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


在 A100 队列上提交 pytorch作业：

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p a100
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

π 超算中已经预置了 `NVIDIA GPU CLOUD <https://ngc.nvidia.com/>`__
提供的优化镜像，通过调用该镜像即可运行 PyTorch
作业，无需单独安装，目前版本为 ``pytorch/1.6.0``\ 。

查看π 超算上已编译的软件模块:

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


查看思源一号上已编译的软件模块:

.. code:: bash
   
   module use /dssg/share/imgs/
   module av pytorch

调用该模块:

.. code:: bash

   module load pytorch/1.6.0

以下 slurm 脚本，在 A100 队列上使用 2 块 gpu，并配比 12 cpu
核心。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p a100
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2

   module use /dssg/share/imgs/
   module load pytorch/1.6.0

   python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test


Pytorch单卡性能测试
---------------------
算例下载：

.. code:: console

   $ git clone https://github.com/SJTU-HPC/HPCTesting.git
   $ cd HPCTesting/pytorch-gpu-benchmark

DGX-2运行脚本：

.. code:: bash

   #!/bin/bash
   #SBATCH -p dgx2
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:1
   #SBATCH -N 1

   singularity  run --nv   /lustre/share/singularity/modules/pytorch/1.6.0.sif python benchmark_models.py --folder v100 -w 10 -n 5  -b 32 -g 1 && &>/dev/null 


A100运行脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH -p a100
   #SBATCH -n 1
   #SBATCH --ntasks-per-node 1
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:1
   #SBATCH -N 1

   singularity  run --nv   /dssg/share/imgs/pytorch/1.6.0.sif python benchmark_models.py --folder a100 -w 10 -n 5  -b 32 -g 1 && &>/dev/null 


DGX-2测试结果将被放在 ``v100`` 文件夹内，A100测试结果将被放在 ``a100`` 文件夹内，均为CSV格式。


单卡测试结果:

resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,wide_resnet101_2模型的平均batch耗时如下：

+--------------------------------------+
| pytorch                              |
+=========+=========+=========+========+
|partition| mode    |precision|ms/batch|
+---------+---------+---------+--------+
| v100    | train   | double  |   471  |
+---------+---------+---------+--------+
| v100    | train   | float   |  180   |
+---------+---------+---------+--------+
| v100    | train   | half    |     91 |
+---------+---------+---------+--------+
| v100    |inference| double  |    148 |
+---------+---------+---------+--------+
| v100    |inference| float   |    63  |
+---------+---------+---------+--------+
| v100    |inference| half    |    26  |
+---------+---------+---------+--------+
| a100    | train   | double  |   350  |
+---------+---------+---------+--------+
| a100    | train   | float   |    78  |
+---------+---------+---------+--------+
| a100    | train   | half    |     60 |
+---------+---------+---------+--------+
| a100    |inference| double  |   107  |
+---------+---------+---------+--------+
| a100    |inference| float   |    25  |
+---------+---------+---------+--------+
| a100    |inference| half    |     16 |
+---------+---------+---------+--------+

Pytorch双卡性能测试
---------------------
算例下载：

.. code:: console

   $ git clone https://github.com/SJTU-HPC/HPCTesting.git
   $ cd HPCTesting/pytorch-gpu-benchmark

DGX-2运行脚本：

.. code:: bash

   #!/bin/bash
   #SBATCH -p dgx2
   #SBATCH -n 2
   #SBATCH --ntasks-per-node 2
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:2
   #SBATCH -N 1

   singularity  run --nv   /lustre/share/singularity/modules/pytorch/1.6.0.sif python benchmark_models.py --folder v100 -w 10 -n 5  -b 16 -g 1 && &>/dev/null 


A100运行脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH -p a100
   #SBATCH -n 2
   #SBATCH --ntasks-per-node 2
   #SBATCH --cpus-per-task 6 
   #SBATCH --gres gpu:2
   #SBATCH -N 1

   singularity  run --nv   /dssg/share/imgs/pytorch/1.6.0.sif python benchmark_models.py --folder a100 -w 10 -n 5  -b 16 -g 1 && &>/dev/null 


DGX-2测试结果将被放在 ``v100`` 文件夹内，A100测试结果将被放在 ``a100`` 文件夹内，均为CSV格式。


双卡测试结果:

resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,wide_resnet101_2模型的平均batch耗时如下：

+--------------------------------------+
| pytorch                              |
+=========+=========+=========+========+
|partition| mode    |precision|ms/batch|
+---------+---------+---------+--------+
| v100    | train   | double  |     289|
+---------+---------+---------+--------+
| v100    | train   | float   |    134 |
+---------+---------+---------+--------+
| v100    | train   | half    |     79 |
+---------+---------+---------+--------+
| v100    |inference| double  |    102 |
+---------+---------+---------+--------+
| v100    |inference| float   |    56  |
+---------+---------+---------+--------+
| v100    |inference| half    |     38 |
+---------+---------+---------+--------+
| a100    | train   | double  |   213  |
+---------+---------+---------+--------+
| a100    | train   | float   |    68  |
+---------+---------+---------+--------+
| a100    | train   | half    |   57   |
+---------+---------+---------+--------+
| a100    |inference| double  |    69  |
+---------+---------+---------+--------+
| a100    |inference| float   |  30    |
+---------+---------+---------+--------+
| a100    |inference| half    |   25   |
+---------+---------+---------+--------+

建议
----------------
A100 显卡的内存为40GB，而V100显卡内存为32GB。从测试结果也可以看出，A100无论是训练还是推理，在单精度、双精度、半精度的计算速度均大幅领先于 V100，推荐大家使用思源一号的 A100队列提交pytorch作业。


参考资料
--------

-  `PyTorch官网 <https://pytorch.org/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
