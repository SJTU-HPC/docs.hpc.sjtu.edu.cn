.. _DELLY:

DELLY
=================

简介
-------------

Delly是一种集成的结构变异（SV）预测方法，可以在短期读取大规模并行测序数据，
以单核苷酸分辨率发现基因分型和可视化缺失、串联重复、倒位和易位等缺陷。它使
用配对末端，拆分阅读和阅读深度来敏感而准确地描绘整个基因组的重排。
关于Delly的更多信息请访问：https://tobiasrausch.com/delly/。

目前ARM超算已经部署了0.8.3版本的delly；π集群上未全局部署，用户可通过conda自行安装。

.. _ARM版本delly:

ARM 版本 delly 
--------------------

在 `ARM 节点 <../login/index.html#arm>`__\ 查看已编译的软件模块：

.. code:: bash

   module avail delly

在 `ARM 节点 <../login/index.html#arm>`__\ 调用已编译的软件模块：

.. code:: bash

   module load delly/0.8.3

示例脚本如下(delly.slurm):

.. code:: bash
   
   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=arm128c256g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load delly/0.8.3
   delly --version
   delly --help

在 `ARM 节点 <../login/index.html#arm>`__\ 上使用如下指令提交（若在 π2.0 登录节点上提交将出错）：

.. code:: bash
  
   sbatch delly.slurm

π集群上conda安装的完整步骤
------------------------------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda delly
