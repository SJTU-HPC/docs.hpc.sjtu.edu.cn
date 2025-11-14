.. _gpumd:

GPUMD
=====

简介
----

GPUMD 是一款在GPU上实现的通用分子动力学软件包。GPUMD可以实现对金属、半导体、有机物等各类材料的高效率和高性价比的分子动力学模拟。GPUMD 提供极为高效NEP机器学习势函数，实现NEP训练与应用的无缝衔接，将量子力学精度的模拟扩展至介观尺度。

从源码编译 GPUMD 方法
------------------------

.. code:: bash

   git clone https://github.com/brucefan1983/GPUMD.git #这一步可以使用 github 的镜像站加速下载
   cd src/
   module load cuda/11.8.0
   make

编译成功后，将在 src 文件夹下生成两个可执行文件 gpumd和nep

运行 GPUMD 方法
----------------

GPUMD 已作为环境模块安装在 π 和思源一号集群上。要运行模拟，请遵循以下步骤。

1. 加载模块

.. code:: bash

   module load gpumd

2. 进入数据目录

.. code:: bash

   cd /path/to/your/data_directory

3. 执行模拟

.. attention::

   严禁在登录节点上直接运行 ``gpumd``。你必须通过 Slurm 调度系统将作业提交到 GPU 计算分区

在 π 上运行
~~~~~~~~~~~

脚本如下 test1.slurm

.. code:: bash

   #!/bin/bash 

   #SBATCH -J gpumd_example 
   #SBATCH -p dgx2 
   #SBATCH -N 1 
   #SBATCH --gres=gpu:1 
   #SBATCH -o %j.out 
   #SBATCH -e %j.err 

   module load gpumd

   # 设置资源限制，防止某些模拟意外终止 
   ulimit -s unlimited 
   ulimit -l unlimited

   # 这里进入你的数据所在文件夹
   EXAMPLE_DIR="~/GPUMD/examples"
   cd $EXAMPLE_DIR

   #  运行gpumd模拟
   srun gpumd

   #  运行nep
   # srun nep

提交作业

.. code:: bash

   $ sbatch test1.slurm

在思源一号上运行
~~~~~~~~~~~~~~~~

脚本如下 test2.slurm

.. code:: bash

   #!/bin/bash 

   #SBATCH -J gpumd_example 
   #SBATCH -p a100 
   #SBATCH -N 1 
   #SBATCH --gres=gpu:1 
   #SBATCH -o %j.out 
   #SBATCH -e %j.err 

   # 设置资源限制，防止某些模拟意外终止 
   ulimit -s unlimited 
   ulimit -l unlimited

   # 这里进入你的数据所在文件夹
   EXAMPLE_DIR="~/GPUMD/example"
   cd $EXAMPLE_DIR

   #  运行gpumd模拟
   srun ~/GPUMD/src/gpumd

   #  运行nep
   srun ~/GPUMD/src/nep

提交作业

.. code:: bash

   $ sbatch test2.slurm

参考资料
--------

-  `GPUMD 官网 <https://gpumd.org/>`__
-  `GPUMD 示例 <https://github.com/brucefan1983/GPUMD/tree/master/examples>`__