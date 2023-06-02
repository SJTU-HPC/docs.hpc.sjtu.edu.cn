ARM 节点使用文档
====================

ARM 节点基于 ARM 处理器构建，是国内首台基于 ARM 处理器的校级超算集群，共 100 个计算节点，与 π 2.0 集群实现共享登录、共享 Lustre 文件系统和共享 Slurm 作业调度系统，完美融入现有超算系统。

ARM 单节点配备 128 核（2.6 GHz）、256 GB 内存（16 通道 DDR4-2933）、240 GB 本地硬盘，节点间采用 IB 高速互联，挂载 Lustre 并行文件系统。

ARM 作业提交方式与 π 2.0 一致。在 Slurm 作业调度系统里，100 个 ARM 计算节点以 \ ``arm128c256g``\ 队列名称统一调度。

ARM 使用须知
------------------------

* ARM 与 π 2.0 的 X86 CPU 指令集不同，在 π 2.0 上使用的软件无法直接在新队列上运行，必须使用 ARM 平台上统一部署的应用，或在 ARM 计算节点上自行重新编译。

* 软件编译和作业提交，均需在 ARM 节点上，不能在 π 2.0 节点上。

ARM 节点登录
------------------

* ARM 配备单独的登录节点，SSH 登录命令如下：

.. code:: bash

	ssh username@armlogin.hpc.sjtu.edu.cn
	
* 也可以从 π 2.0 上登录到 ARM 计算节点：

.. code:: bash

	srun -p arm128c256g -n 4 --pty /bin/bash


ARM 应用支持
------------------

由于 CPU 架构不同，原 π 2.0 的应用软件都需要重新编译。我们已在 ARM 集群上部署了首批主流计算软件。后续将会推出更多的适用 ARM 集群运行的应用。

* 应用查看：(在 ARM 登录节点或计算节点) \ ``module av``\ 命令；

* 应用加载：(在 ARM 计算节点) \ ``module load``\ 命令；

	
ARM 作业示例
------------------

以下是一个名为\ ``arm.slurm``\ 的 **单节点** 作业脚本，该脚本申请1个ARM节点（128核）。

.. code:: bash

	#!/bin/bash

	#SBATCH --job-name=test       
	#SBATCH --partition=arm128c256g       
	#SBATCH -N 1           
	#SBATCH --ntasks-per-node=128
	#SBATCH --output=%j.out
	#SBATCH --error=%j.err

	module load XXX

	mpirun -n $SLURM_NTASKS ...

用以下方式提交作业（请注意，ARM 作业务必在 ARM 的登录节点或计算节点提交）：

.. code:: bash

   $ sbatch arm.slurm

``squeue``\ 可用于检查作业状态。


有两点需注意：

* 并行命令采用 mpirun，暂不推荐 srun

* 暂不限制节点共享或独占（不区分是否是 small 类型的作业）



ARM 交互作业示例
~~~~~~~~~~~~~~~~~~~~~~~~

``srun``\ 可以启动交互式作业。该操作将阻塞，直到完成或终止。启动远程主机 bash 终端的命令：

.. code:: bash

   $ srun -p arm128c256g -n 4 --pty /bin/bash


ARM 集群软件编译
~~~~~~~~~~~~~~~~~~~~~~~~

建议大家优先使用我们提供的软件，若需要自行编译软件，请登录到 ARM 的计算节点，按照软件文档操作。目前 ARM HPC 应用生态正在逐步完善中，建议选择软件的最新版本，并了解其对 ARM 的支持情况。
