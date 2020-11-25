****
容器
****

Pi 集群支持用户使用 Singularity 方法构建自己的容器应用。Pi 上许多应用软件也是通过 Singularity 方法安装的。

在集群上使用Singularity
=======================

高性能容器Singularity
---------------------

Singularity是劳伦斯伯克利国家实验室专门为大规模、跨节点HPC和DL工作负载而开发的容器化技术。具备轻量级、快速部署、方便迁移等诸多优势，且支持从Docker镜像格式转换为Singularity镜像格式。与Docker的不同之处在于：

1. Singularity同时支持root用户和非root用户启动，且容器启动前后，用户上下文保持不变，这使得用户权限在容器内部和外部都是相同的。
2. Singularity强调容器服务的便捷性、可移植性和可扩展性，而弱化了容器进程的高度隔离性，因此量级更轻，内核namespace更少，性能损失更小。

本文将向大家介绍在集群上使用Singularity的方法。

如果我们可以提供任何帮助，请随时联系\ `hpc邮箱 <hpc@sjtu.edu.cn>`__\ 。

镜像准备
--------

首先我们需要准备Singularity镜像。如果镜像来自于\ `Docker
Hub <https://hub.docker.com/>`__\ ，则可以直接在集群中使用如下命令制作镜像。

.. code:: bash

   $ singularity build ubuntu.simg docker://ubuntu
   INFO:    Starting build...
   Getting image source signatures
   ...
   INFO:    Creating SIF file...
   INFO:    Build complete: ubuntu.simg

如果需要自行构建镜像或者修改现有镜像，因为其过程需要root权限，我们建议:

1. 使用交大高性能计算中心自研的U2BC非特权用户容器构建服务，参见\ `非特权用户容器构建 <../u2cb>`__\ 。
2. 使用个人的Linux环境进行镜像构建然后传至集群。

我们在集群中预置了以下软件的Singularity的镜像。

======== ========================================
软件     位置
======== ========================================
PyTorch  /lustre/share/img/pytorch-19.10-py3.simg
Gromacs  /lustre/share/img/gromacs-2018.2.simg
vmd      /lustre/share/img/vmd-1.9.3.simg
octave   /lustre/share/img/octave-4.2.2.simg
openfoam /lustre/share/img/openfoam-6.simg
======== ========================================

任务提交
--------

可以通过作业脚本然后使用\ ``sbatch``\ 进行作业提交，以下示例为在DGX-2上使用PyTorch的容器作业脚本示例，其中作业使用单节点并分配2块GPU：

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

   IMAGE_PATH=/lustre/share/img/pytorch-19.10-py3.simg

   singularity run --nv $IMAGE_PATH python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'

我们假设这个脚本文件名为\ ``pytorch_singularity.slurm``,使用以下指令提交作业。

.. code:: bash

   $ sbatch pytorch_singularity.slurm

交互式提交
----------

.. code:: shell

   srun -n 1 -p dgx2 --gres=gpu:2 --pty singularity shell --nv /lustre/share/img/pytorch-19.10-py3.simg
   Singularity pytorch-19.10-py3.simg:~/u2cb_test> python -c "import torch;print(torch.__version__)"
   1.3.0a0+24ae9b5






非特权用户容器构建
==================

U2CB是上海交通大学高性能计算中心自行研发的非特权用户容器构建平台。在集群上普通用户可以使用U2CB自行构建Singularity镜像。

容器构建流程
------------

镜像创建
~~~~~~~~

支持从\ `Docker Hub <https://hub.docker.com/>`__\ 或者\ `NVIDIA
NGC <https://ngc.nvidia.com/>`__\ 提供的镜像开始构建。如下指令，从\ ``docker://ubuntu:latest``\ 构建名为\ ``ubuntu-test``\ 的镜像。从\ ``docker://nvcr.io/nvidia/pytorch:20.02-py3``\ 构建名为\ ``pytorch-test``\ 的镜像。

.. code:: shell

   $ u2cb create -n ubuntu-test -b docker://ubuntu:latest
   $ u2cb create -n pytorch-test -b docker://nvcr.io/nvidia/pytorch:20.02-py3

从定义文件构建镜像创建（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以参考Singularity的\ `Definition
Files <https://sylabs.io/guides/3.5/user-guide/definition_files.html>`__\ 编写您的定义文件。

例如，在您的本地编辑定义文件\ ``test.def``\ ，内容为：

::

   Bootstrap: docker
   From: ubuntu

   %post
       apt update && apt install -y gcc

   %enviroment
       export TEST_ENV_VAR=SJTU

然后使用u2cb进行镜像构建：

.. code:: shell

   $ u2cb defcreate -n ubuntu-test -d ./test.def

镜像查询
~~~~~~~~

完成镜像创建后，可以使用如下指令进行镜像查询。

.. code:: shell

   $ u2cb list
   ubuntu-test pytorch-test

与镜像进行交互
~~~~~~~~~~~~~~

如需要与镜像进行交互，可以使用如下指令连接至容器中，在容器中可以使用root权限进行软件安装等特权行为，
以ubuntu为例，比如\ ``apt install``\ ：

.. code:: shell

   $ u2cb connect -n ubuntu-test
   Singularity> whoami
   root
   Singularity> apt update && apt install -y gcc

注意事项：

1. 请勿将任何应用安装在\ ``/root``\ 下（因容器在集群上运行时为普通用户态，\ ``/root``\ 不会被打包），推荐直接安装在系统目录或者\ ``/opt``\ 下；

2. 运行应用所需的环境变量可以添加到\ ``/enviroment``\ 文件中。

.. code:: shell

   Singularity> echo "export TEST_ENV_VAR=SJTU" >> /environment         
   Singularity> echo "export PATH=/opt/app/bin:$PATH" >> /environment

镜像下载
~~~~~~~~

可以使用如下指令可以将镜像从构建服务器上打包并下载到本地\ ``./ubuntu-test.simg``\ ，然后可以在集群环境中使用该镜像，详细可见\ `容器 <../singularity/#_2>`__\ 一节。

.. code:: shell

   $ u2cb download -n ubuntu-test
   $ srun -p small -n 1 --pty singularity shell ubuntu-test.simg

镜像删除
~~~~~~~~

使用如下指令删除在构建服务器上的镜像文件。

.. code:: shell

   $ u2cb delete -n ubuntu-test

U2CB Shell
~~~~~~~~~~

U2CB还支持用户通过\ ``u2cb shell``\ 登录U2CB
Server，进行镜像查询，镜像交互，镜像删除的功能。

.. code:: shell

   $ u2cb shell
   (U2CB Server) > help

   Documented commands (type help <topic>):
   ========================================
   create  delete  help  list  shell

   (U2CB Server) > help list

           Use `list` to see all containers
           Use `list def` to see all define files
           Use `list img` to see all image files

   (U2CB Server) > list def

参考资料
--------

-  `Singularity Quick
   Start <https://sylabs.io/guides/3.4/user-guide/quick_start.html>`__
-  `Docker Hub <https://hub.docker.com/>`__
-  `NVIDIA GPU CLOUD <https://ngc.nvidia.com/>`__
-  `Fakeroot feature of
   Singularity <https://sylabs.io/guides/3.5/user-guide/fakeroot.html>`__

   
