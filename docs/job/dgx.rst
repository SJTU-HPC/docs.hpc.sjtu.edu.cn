AI平台使用文档
================

交大AI计算平台是国内高校计算力能最强的人工智能计算平台。AI平台拥有两种 GPU 计算资源：

* A100：思源一号 `a100` 队列含 23 个 GPU 计算节点，每个节点为 NVIDIA HGX A100 4-GPU。每块 A100 默认配置 16 个 CPU 核心。

* DGX-2：Pi 上的 `dgx2` 队列含 DGX-2 8 台，每台 DGX-2 配备 16 块 NVIDIA Tesla V100，每块 V100 默认配置 6 个 CPU 核心。通过搭载NVIDIA NVSwitch创新技术， GPU间带宽高达 2.4 TB/s。

本文档将介绍两种 GPU 使用方法（作业提交模式、交互模式）及 GPU 利用率查看方法。

A100 使用
-------------

AI 计算平台有两种使用 GPU 的形式：作业提交模式、交互模式。作业提交模式适合正式运行作业，交互模式适用于调试和安装。

作业提交模式
^^^^^^^^^^^^^^^^^^^^

提交 `a100` 队列作业请使用 **思源一号登录节点**。

这是一个 **单机单卡** 作业脚本，该脚本向 `a100` 队列申请 1 块 GPU（默认配置 16 个 CPU 核心），并在作业完成时邮件通知。此示例作业中执行的为 NVIDIA Sample中的 \ ``cudaTensorCoreGemm``\ 。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1 
   #SBATCH --cpus-per-task=16
   #SBATCH --gres=gpu:1            #若使用 2 块卡，就给 gres=gpu:2
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc cuda

   ./cudaTensorCoreGemm

A100 交互模式
-------------

使用如下命令申请一卡 GPU，默认配置 16 CPU 核心。该交互模式适用于调试和安装：

.. code:: bash

   $ srun -p dgx2 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=6 --pty /bin/bash
   $ module load cuda


DGX-2 使用
-------------

作业提交模式
^^^^^^^^^^^^^^^^^^^^

提交 `dgx2` 队列作业请使用 **Pi 登录节点**。

这是一个 **单机单卡** 作业脚本，该脚本向 `dgx2` 队列申请 1 块 GPU（默认配置 6 个 CPU 核心），并在作业完成时邮件通知。此示例作业中执行的为 NVIDIA Sample中的 \ ``cudaTensorCoreGemm``\ 。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1 
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:1              #若使用 2 块卡，就给 gres=gpu:2
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc cuda

   ./cudaTensorCoreGemm

DGX-2 交互模式
-------------

使用如下命令申请一卡 GPU，默认配置 6 CPU 核心。该交互模式适用于调试和安装：

.. code:: bash

   $ srun -p dgx2 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=6 --pty /bin/bash
   $ module load cuda


GPU 利用率查看
------------------

GPU 利用率查看，需先登录正在使用的 GPU 计算节点，然后输入 `nvidia-smi` 查看

以 A100 为例：

.. code:: bash

   $ squeue       # 查看正在计算的 GPU 节点名字，如 gpu03
   $ ssh gpu03    # 登录节点
   $ nvidia-smi





参考资料
--------

-  `DGX-2 User
   Guide <https://docs.nvidia.com/dgx/pdf/dgx2-user-guide.pdf>`__
-  `SLURM Workload Manager <http://slurm.schedmd.com>`__
