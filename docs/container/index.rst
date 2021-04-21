****
容器
****

π 集群支持用户使用 Singularity 方法构建自己的容器应用。π 集群上许多应用软件也是通过 Singularity 方法安装的。

在 π 集群上使用Singularity
=============================

高性能容器Singularity
---------------------

Singularity 是劳伦斯伯克利国家实验室专门为大规模、跨节点HPC和DL工作负载而开发的容器化技术。具备轻量级、快速部署、方便迁移等诸多优势，且支持从Docker镜像格式转换为Singularity镜像格式。与Docker的不同之处在于：

1. Singularity 同时支持root用户和非root用户启动，且容器启动前后，用户上下文保持不变，这使得用户权限在容器内部和外部都是相同的。
2. Singularity 强调容器服务的便捷性、可移植性和可扩展性，而弱化了容器进程的高度隔离性，因此量级更轻，内核namespace更少，性能损失更小。

本文将向大家介绍在 π 集群上使用Singularity的方法。

如果我们可以提供任何帮助，请随时联系\ `hpc邮箱 <hpc@sjtu.edu.cn>`__\ 。

镜像准备
--------

首先我们需要准备Singularity镜像。如果镜像来自于\ `Docker
Hub <https://hub.docker.com/>`__\ ，则可以直接在 π 集群中使用如下命令制作镜像。

.. code:: bash

   $ singularity build ubuntu.simg docker://ubuntu
   INFO:    Starting build...
   Getting image source signatures
   ...
   INFO:    Creating SIF file...
   INFO:    Build complete: ubuntu.simg

如果需要自行构建镜像或者修改现有镜像，因为其过程需要root权限，我们建议:

1. 使用交大高性能计算中心自研的U2BC非特权用户容器构建服务，参见\ `非特权用户容器构建 <../u2cb>`__\ 。
2. 使用个人的Linux环境进行镜像构建然后传至 π 集群。

我们在 π 集群中预置了以下软件的Singularity的镜像。

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

U2CB是上海交通大学高性能计算中心自行研发的非特权用户容器构建平台。在 π 集群上普通用户可以使用U2CB自行构建Singularity镜像。

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

1. 请勿将任何应用安装在\ ``/root``\ 下（因容器在 π 集群上运行时为普通用户态，\ ``/root``\ 不会被打包），推荐直接安装在系统目录或者\ ``/opt``\ 下；

2. 运行应用所需的环境变量可以添加到\ ``/enviroment``\ 文件中。

.. code:: shell

   Singularity> echo "export TEST_ENV_VAR=SJTU" >> /environment         
   Singularity> echo "export PATH=/opt/app/bin:$PATH" >> /environment

镜像下载
~~~~~~~~

可以使用如下指令可以将镜像从构建服务器上打包并下载到本地\ ``./ubuntu-test.simg``\ ，然后可以在 π 集群环境中使用该镜像，详细可见\ `容器 <../singularity/#_2>`__\ 一节。

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


enroot使用说明
==============

enroot是英伟达公司出的一款开源镜像构建/交互工具。在超算上使用enroot，普通用户不再需要特殊权限即可完成镜像的构建与修改。

目前enroot只在GPU节点上进行了部署。如需使用enroot，请先申请GPU节点资源。如只需进行镜像制作，可申请交互式作业，只需在 `srun` 命令中指定 `dgx2` 队列以及使用 `--gres` 选项指定所需的GPU数量。在 `dgx2` 上申请一个交互式作业的示例如下：

.. code:: console
    
    $srun -p dgx2 -n 1 --gres=gpu:1 --pty /bin/bash

更多关于如何申请GPU节点资源的文档请参考 :doc: `DGX-2使用文档 <../job/dgx.rst>`  

创建镜像
--------

从镜像库中下载基础镜像：

.. code:: console

   $ enroot import 'docker://centos:8'

该命令会在当前路径下生成一个\ ``.sqsh``\ 文件：

.. code:: console

   $ ls
   centos+8.sqsh

根据该文件创建镜像：

.. code:: console

   $ enroot create --name centos8 centos+8.sqsh 

其中\ ``--name``\ 后面的参数为自定义的镜像名，我们这里取为”centos8“。
上述命令会在\ ``~/.local/share/enroot/``\ 路径下生成相应的文件夹：

.. code:: sh

   $ ls ~/.local/share/enroot/
   centos8

使用\ ``enroot list``\ 命令列出已经创建的镜像名：

.. code:: console

   $ enroot list
   centos8

启动镜像
--------

使用普通用户启动镜像
~~~~~~~~~~~~~~~~~~~~

使用\ ``enroot start``\ 命令启动镜像：

.. code:: console

   $ enroot start --rw centos8

此命令会以当前用户名进入镜像内的根目录：

.. code:: console

   $ pwd
   /
   $ whoami 
   YOUR_USERNAME

我们可以看到根目录下的所有路径，这些路径与原先的文件目录是分离的:

.. code:: console

   $ ls
   bin  etc   lib    lost+found  media  opt   root  sbin  sys  usr
   dev  home  lib64  lustre      mnt    proc  run   srv   tmp  var

普通用户可以访问、更改这些目录。

使用\ ``exit``\ 命令退出镜像：

.. code:: console

   $ exit
   $ 

使用root用户启动镜像
~~~~~~~~~~~~~~~~~~~~

使用root用户启动镜像，只需在\ ``enroot start``\ 命令中加入\ ``--root``\ 选项：

.. code:: console

   $ enroot start --rw --root centos8
   # whoami
   root

此时可以使用root来运行大部分需要root权限的命令：

.. code:: console

   # yum install python3
   ...
   Installed:
     platform-python-pip-9.0.3-18.el8.noarch                                       
     python3-pip-9.0.3-18.el8.noarch                                               
     python3-setuptools-39.2.0-6.el8.noarch                                        
     python36-3.6.8-2.module_el8.3.0+562+e162826a.x86_64                           

   Complete!

   # python3
   Python 3.6.8 (default, Aug 24 2020, 17:57:11) 
   [GCC 8.3.1 20191121 (Red Hat 8.3.1-5)] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> 

使用镜像执行命令
~~~~~~~~~~~~~~~~

我们可以在镜像中安装各种命令，在需要使用该命令的时候，使用\ ``enroot start --rw image_name command``\ 来调用该命令。

调用结果是在\ **镜像环境**\ 中调用该命令的输出。

.. code:: console

   $ enroot start --rw centos8 ls
   bin  etc   lib    lost+found  media  opt   root  sbin  sys  usr
   dev  home  lib64  lustre      mnt    proc  run   srv   tmp  var
   $ enroot start --rw centos8 python3
   Python 3.6.8 (default, Aug 24 2020, 17:57:11) 
   [GCC 8.3.1 20191121 (Red Hat 8.3.1-5)] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> 

如果我们想用镜像环境中的python来跑一个本地的程序，由于镜像环境中的文件路径与本地不互通，因此python无法找到本地的\ ``.py``\ 文件：

.. code:: console

   $ enroot start --rw centos8 python3 hello.py
   python3: can't open file 'hello.py': [Errno 2] No such file or directory

我们可以通过挂载home目录来解决这个问题。

挂载本地home目录
----------------

.. code:: console

   $ export ENROOT_MOUNT_HOME=y
   $ enroot start --rw centos8
   $ cd ~
   $ ls
   anaconda3      nvidia+cuda+11.1.1-base-ubuntu20.04.sqsh  paraview5.6    test.py
   centos+8.sqsh  ondemand                  singularities  work

此时，提供\ ``.py``\ 文件的路径，即可使用镜像环境运行python程序：

.. code:: sh

   $ enroot start --rw centos8 python3 ~/singularities/hello.py
   hello world!

如需分离本地home目录，只要重置\ ``ENROOT_MOUNT_HOME``\ 变量，重新启动镜像即可：

.. code:: console

   $ unset ENROOT_MOUNT_HOME
   $ enroot start --rw centos8
   $ cd ~
   $ ls
   $

在脚本中使用镜像
----------------

在脚本中使用镜像与在命令行中使用镜像一样。只要使用\ ``enroot start --rw image_name command``\ 即可在镜像环境中调用命令。

示例脚本如下：

.. code:: bash


   #!/bin/bash

   #SBATCH --job-name=dgx2_test
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   export ENROOT_MOUNT_HOME=y
   enroot start --rw centos8 python3 ~/singularities/hello.py

将该脚本保存为\ ``hello.slurm``\ ，使用\ ``sbatch``\ 命令提交作业脚本：

.. code:: console

   $ sbatch hello
   Submitted batch job 4620199

打开输出文件即可看到程序的输出：

.. code:: console

   $ cat 4620199.out 
   hello world!

参考资料
--------

-  `Singularity Quick
   Start <https://sylabs.io/guides/3.4/user-guide/quick_start.html>`__
-  `Docker Hub <https://hub.docker.com/>`__
-  `NVIDIA GPU CLOUD <https://ngc.nvidia.com/>`__
-  `Fakeroot feature of
   Singularity <https://sylabs.io/guides/3.5/user-guide/fakeroot.html>`__
-  `enroot官方文档 <https://github.com/NVIDIA/enroot>`__

   
