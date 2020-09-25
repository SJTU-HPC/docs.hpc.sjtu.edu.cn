
Relion
======

简介
----

Relion 是由 MRC 的 Scheres 在 2012
年发布的针对单颗粒冷冻电镜图片进行处理的框架。

Pi上的Relion
--------------

Pi 上安装有 GPU 版 和 Singularity 版本的 Relion

使用GPU版的Relion
--------------------

GPU Relion模块调用
~~~~~~~~~~~~~~~~~~~

查看 Pi 上已编译的 GPU 版软件:

.. code:: bash

   $ module avail relion

调用该模块:

.. code:: bash

   $ module load relion/3.0.8-gcc-8.3.0-openmpi

GPU Relion的Slurm 脚本
~~~~~~~~~~~~~~~~~~~~~~~~

在 dgx2 队列上使用 1 块 gpu，并配比 6 cpu 核心

.. code:: bash

   #!/bin/bash

   #SBATCH -J relion
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 6 # number of tasks
   #SBATCH --ntasks-per-node=6
   #SBATCH --gres=gpu:1

   module purge
   module load cuda/9.0.176-gcc-4.8.5
   module load openmpi/3.1.5-gcc-9.2.0
   module load relion/3.0.8-gcc-8.3.0-openmpi

   srun --mpi=pmi2 relion_refine_mpi (relion 的命令...)

GPU Relion提交作业
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   $ sbatch slurm.test

使用Relion容器镜像
--------------------

集群中已预置了编译优化的容器镜像，通过调用该镜像即可运行
Relion，无需单独安装，目前版本为 ``relion-3.0.8``\ 。该容器文件位于
``/lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg``

使用singularity容器提交Relion作业
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

示例：在 DGX-2 上使用 Relion 容器，作业使用单节点并分配 2 块 GPU：

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --mem=MaxMemPerNode
   #SBATCH --gres=gpu:2

   IMAGE_PATH=/lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg

   singularity run --nv $IMAGE_PATH relion_refine_mpi --version

假设这个脚本文件名为
``relion_singularity.slurm``\ ，使用以下指令提交作业

.. code:: bash

   $ sbatch relion_singularity.slurm

使用HPC Studio启动可视化界面
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

参照\ `可视化平台 <../../login/HpcStudio/>`__\ ，登陆 HPC
Studio，在顶栏选择 Relion：

|avater| |image1|

参考资料
--------

-  `Relion 官网 <http://www2.mrc-lmb.cam.ac.uk/relion>`__
-  `Singularity 文档 <https://sylabs.io/guides/3.5/user-guide/>`__

.. |avater| image:: ../img/relion2.png
.. |image1| image:: ../img/relion1.png
