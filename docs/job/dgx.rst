GPU 节点使用文档
================

交我算HPC+AI平台拥有两种 GPU 节点：

* a100 (思源一号)：包含 23 个 GPU 计算节点，每个节点为 NVIDIA HGX A100 4-GPU，每块 A100 默认配置 16 个 CPU 核心。

* DGX-2 (AI 计算平台）：AI 平台的 `dgx2` 队列含 DGX-2 8 台，每台 DGX-2 配备 16 块 NVIDIA Tesla V100，每块 V100 默认配置 6 个 CPU 核心。通过搭载NVIDIA NVSwitch创新技术， GPU间带宽高达 2.4 TB/s。

本文档将介绍两种 GPU 节点使用方法（作业提交模式、交互模式）及 GPU 利用率查看方法。

a100 节点
-------------

基于a100计算节点，平台提供两种 GPU 队列：用于正式计算的a100队列，用于调试的debuga100队列。

a100队列
^^^^^^^^^^^^^^^^^^^^^^^^

提交a100队列作业请使用 **思源一号登录节点**。

这是一个 **单机单卡** 作业脚本，该脚本向 `a100` 队列申请 1 块 GPU（默认配置 16 个 CPU 核心），并在作业完成时邮件通知。此示例作业中执行的为 NVIDIA Sample中的 \ ``cudaTensorCoreGemm``\ 。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test
   #SBATCH --partition=a100
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=16
   #SBATCH --gres=gpu:1            #若使用 2 块卡，就给 gres=gpu:2
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc cuda

   ./cudaTensorCoreGemm

或者也可以申请节点资源进行交互操作，使用如下命令申请一卡 GPU，默认配置 16 CPU 核心：

.. code:: bash

   $ srun -p a100 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=16 --pty /bin/bash
   $ module load cuda

.. tip::

   在登录节点 `srun` 执行交互作业时可能会断连导致作业中断，建议在 `HPC Studio <https://studio.hpc.sjtu.edu.cn/>`_ 申请1核心的远程桌面（cpu节点即可），选择好时间，在计算节点来执行 `srun`。

debuga100队列
^^^^^^^^^^^^^^^^^^^^^^^^

提交debuga100队列作业请使用 **思源一号登录节点**。

debuga100队列是 **调试用的队列** ，目前只提供1节点，因此只能投递单节点作业。调试节点将4块a100物理卡虚拟成4*7=28块gpu卡，每卡拥有5G独立显存；节点CPU资源依然为64核，请在作业参数中合理指定gpu与cpu的配比。

投放到此队列的作业 **运行时间最长为20分钟** ，超时后会被终止。

调试作业脚本示例：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test
   #SBATCH --partition=debuga100
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=25
   #SBATCH --gres=gpu:5            #最多28gpu卡
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc cuda

   ./cudaTensorCoreGemm


DGX-2 节点
-------------

基于DGX-2计算节点，平台提供dgx2计算队列用于正式计算。

dgx2队列
^^^^^^^^^^^^^

提交dgx2队列作业请使用 **π 2.0 集群登录节点**。

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

或者也可以申请节点资源进行交互操作，使用如下命令申请一卡 GPU，默认配置 6 CPU 核心：

.. code:: bash

   $ srun -p dgx2 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=6 --pty /bin/bash
   $ module load cuda


GPU 利用率查看
------------------

GPU 利用率查看，需先登录正在使用的 GPU 计算节点，然后输入 `nvidia-smi` 查看

以 a100 为例：

.. code:: bash

   $ squeue       # 查看正在计算的 GPU 节点名字，如 gpu03
   $ ssh gpu03    # 登录节点
   $ nvidia-smi



参考资料
-----------

-  `DGX-2 User
   Guide <https://docs.nvidia.com/dgx/pdf/dgx2-user-guide.pdf>`__
-  `SLURM Workload Manager <http://slurm.schedmd.com>`__
