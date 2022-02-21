思源一号使用文档
==============

杨元庆科学计算中心“思源一号”高性能计算集群总算力 6 PFLOPS（每秒六千万亿次浮点运算），是目前国内高校第一的超算集群。

CPU 采用双路 Intel Xeon ICX Platinum 8358 32 核，主频 2.6GHz，共 936 个计算节点；GPU 采用 NVIDIA HGX A100 4-GPU，

共 23 个计算节点。计算节点之间使用 Mellanox 100 Gbps Infiniband HDR 高速互联，并行存储的聚合存储能力达 10 PB。

思源一号为独立集群，使用dssg文件系统，采用SLURM作业调度，提交方式与π 2.0一致，CPU和GPU队列名分别为64c512g和a100。

思源一号 使用须知
------------------------

* ARM 超算与 π 2.0 的 X86 CPU 指令集不同，在 π 2.0 上使用的软件无法直接在新队列上运行，必须使用 ARM 平台上统一部署的应用，或在 ARM 计算节点上自行重新编译。

* 软件编译和作业提交，均需在 ARM 节点上，不能在 π 2.0 节点上。

思源一号 登录
------------------

* 思源一号配备单独的登录节点，SSH 登录命令如下：

.. code:: bash

   $ ssh username@sylogin1.hpc.sjtu.edu.cn



思源一号 应用支持
------------------

思源一号为独立集群，部署的软件和编译器版本与π 2.0不同

* 应用查看：(在 思源一号登录节点或计算节点) \ ``module av``\ 命令；

* 应用加载：(在 思源一号计算节点) \ ``module load``\ 命令；


思源一号 作业示例
------------------

以下是一个名为\ ``siyuan.slurm``\ 的 **单节点** 作业脚本，该脚本申请1个思源一号CPU节点（64核）。

.. code:: bash

	#!/bin/bash

	#SBATCH --job-name=test
	#SBATCH --partition=64c512g
	#SBATCH -N 1
	#SBATCH --ntasks-per-node=64
	#SBATCH --output=%j.out
	#SBATCH --error=%j.err

	module load XXX

	mpirun -n $SLURM_NTASKS ...

用以下方式提交作业（请注意，思源一号作业请在思源一号的登录节点或计算节点提交）：

.. code:: bash

   $ sbatch siyuan.slurm

``squeue``\ 可用于检查作业状态。




思源一号交互作业示例
~~~~~~~~~~~~~~~~~~~~~~~~

``srun``\ 可以启动交互式作业。该操作将阻塞，直到完成或终止。启动远程主机 bash 终端的命令：

.. code:: bash

   $ srun -p 64c512g -n 4 --pty /bin/bash
