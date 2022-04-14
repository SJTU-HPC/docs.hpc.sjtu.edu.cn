AI平台使用文档
=============

交大AI计算平台是国内高校计算力能最强的人工智能计算平台。该平台有用A100和V100两种类型的显卡。Pi超算GPU资源由8台DGX-2组成，每台DGX-2配备16块NVIDIA
Tesla V100，深度学习张量计算能力可以达到16PFLOPS；通过搭载NVIDIA
NVSwitch创新技术， GPU间带宽高达 2.4
TB/s。思源一号GPU 采用 NVIDIA HGX A100 4-GPU，共 23 个计算节点。AI计算平台采用可扩展架构，使得模型的复杂性和规模不受传统架构局限性的限制，从而可以应对众多复杂的人工智能挑战。

AI 计算平台每块 V100 默认配置 6 个 CPU 核心。

本文将向大家介绍DGX-2的使用方法。

如果我们可以提供任何帮助，请随时联系\ `hpc邮箱 <hpc@sjtu.edu.cn>`__\ 。

DGX-2作业示例
-------------

sbatch提交示例
~~~~~~~~~~~~~~

通过\ ``sbatch``\ 命令提交作业是最推荐的用法。

.. code:: bash

   $ sbatch jobscript.slurm

Slurm具有丰富的参数集。 以下最常用的。

========================= ==================
Slurm                     含义
========================= ==================
-n [count]                总进程数
–ntasks-per-node=[count]  每台节点上的进程数
-p [partition]            作业队列
–job-name=[name]          作业名
–gres=gpu:[gpus]          gpu数量
–output=[file_name]       标准输出文件
–error=[file_name]        标准错误文件
–time=[dd-hh:mm:ss]       作业最大运行时长
–exclusive                独占节点
-mail-type=[type]         通知类型
–mail-user=[mail_address] 通知邮箱
–nodelist=[nodes]         偏好的作业节点
–exclude=[nodes]          避免的作业节点
–depend=[state:job_id]    作业依赖
–array=[array_spec]       序列作业
========================= ==================

这是一个名为\ ``dgx.slurm``\ 的 **单机单卡**
作业脚本，该脚本向dgx2队列申请1块GPU（默认配置6个CPU核心），并在作业完成时通知。在此示例作业中执行的为NVIDIA
Sample中的\ ``cudaTensorCoreGemm``\ 。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=dgx2_test
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1 
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:1
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc cuda

   ./cudaTensorCoreGemm

或者通过如下脚本提交 **单机多卡** 作业。

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=dgx2_test
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=4
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:4
   #SBATCH --mail-type=end
   #SBATCH --mail-user=YOU@EMAIL.COM
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc cuda

   ./cudaTensorCoreGemm

用以下方式提交作业：

.. code:: bash

   $ sbatch dgx.slurm

``squeue``\ 可用于检查作业状态。用户可以在作业执行期间通过SSH登录到计算节点。输出将实时更新到文件[jobid]
.out和[jobid] .err。

srun提交示例
~~~~~~~~~~~~

``srun``\ 可以启动交互式作业。该操作将阻塞，直到完成或终止。例如，在DGX-2上运行\ ``hostname``\ 。

.. code:: bash

   $ srun -N 1 -n 1 -p dgx2 --gres=gpu:2 hostname
   vol01

启动远程主机bash终端。

.. code:: bash

   $ srun -N 1 -n 1 -p dgx2 --gres=gpu:1 --pty /bin/bash
   $ hostname
   vol01

GPU程序调试
-----------

启动远程主机bash终端，然后使用cuda toolkit中提供的cuda-gdb工具调试程序。

.. code:: bash

   $ srun -N 1 -n 1 -p dgx2 --gres=gpu:1 --pty /bin/bash
   $ module load cuda
   $ cuda-gdb ./your_app

参考资料
--------

-  `DGX-2 User
   Guide <https://docs.nvidia.com/dgx/pdf/dgx2-user-guide.pdf>`__
-  `SLURM Workload Manager <http://slurm.schedmd.com>`__
