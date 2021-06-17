****
容器
****

容器是一种Linux上广为采用的应用封装技术，它将可执行程序与依赖库打包成一个镜像文件，启动时与宿主节点共享操作系统内核。
由于镜像文件同时携带可执行文件和依赖库，避免了两者不匹配造成的兼容性问题，还能在一个宿主Linux操作系统上支持多种不同的Linux发行版，譬如在CentOS发行版上运行Ubuntu的 ``apt-get`` 命令。

π 超算集群采用基于 `Singularity <https://sylabs.io/singularity/>`__  的高性能计算容器技术，相比Docker等在云计算环境中使用的容器技术，Singularity 同时支持root用户和非root用户启动，且容器启动前后，用户上下文保持不变，这使得用户权限在容器内部和外部都是相同的。
此外，Singularity 强调容器服务的便捷性、可移植性和可扩展性，而弱化了容器进程的高度隔离性，因此量级更轻，内核namespace更少，性能损失更小。

您选择如下其中一种方法使用集群提供的Singularity容器镜像能力：

1. 使用Singularity加载集群预编译的镜像。
2. 使用Singularity拉取远端镜像。
3. 从零开始按需定制Singularity镜像。

.. TODO: 使用Singularity加载集群预编译的镜像
.. TODO: ===================================
.. TODO: 胡筱婧

使用Singularity加载远端镜像
---------------------------

您可以使用 ``singularity pull`` 拉取远端预编译的镜像，从而直接使用外部预预编译镜像仓库提供的丰富软件资源。
Singularity可以从Docker Hub(以 ``docker://`` 开头)、Singularity Hub(以 ``shub://`` 开头)等地址拉取镜像。
如下命令从Docker Hub拉取CentOS 8标准镜像，保存成名为 ``cento8.sif`` 的镜像文件::

.. code:: bash

    $ singularity pull centos8.sif docker://centos:centos8

查看生成的镜像文件

.. code:: bash

    $ ls centos8.sif
    centos8.sif

加载容器镜像，并且在容器环境中运行 ``cat`` 程序，查看容器内 ``/etc/redhat-release`` 文件的内容，然后在宿主环境中运行同样命令，对比结果::

.. code:: bash
    $ singularity exec centos.sif cat /etc/redhat-release
    CentOS Linux release 8.3.2011
    $ cat /etc/redhat-release
    CentOS Linux release 7.7.1908 (Core)

运行结果显示，我们成功在CentOS 7操作系统上加载了一个CentOS 8容器镜像。

.. tip:: Singularity镜像文件(Singularity Image File, sif)是一种内容只读的文件格式，其文件内容不能被修改。

从零开始按需定制Singularity镜像
-------------------------------

Singularity使用“镜像定义文件”(Definition File)描述镜像构建过程。
镜像定义文是一个文本文件，描述了构建镜像使用的基本镜像、构建过程执行的命令，其中不少命令通常需要root特权，如软件包管理命令 ``yum``, ``apt-get`` 等等。
运行Singularity容器构建命令 ``singularity build`` ，以及构建过程中在镜像内部的系统相关命令，都需要root特权，因此构建容器的操作通常在您自管的、具有root特权的计算机上完成。
在HPC等共享集群环境中，普通用户没有root权限，因而无法在集群上定制和构建镜像。


在π超算集群上，我们采用“容器化的Singularity”，允许用户在一个受限的环境内以普通用户身份“模拟”root特权，编写自定义的镜像文件，从定义文件构建镜像，并将镜像传回集群使用。

.. tip:: 构建Singularity容器镜像通常需要root特权，通常超算集群不支持这样的操作。π超算集群的“容器化的Singularity”允许用户编写、构建和传回自定义容器镜像。

首先从登录节点使用用户名 ``build`` 跳转到专门用于构建容器镜像的节点。
需要注意的是，X86节点(用于 ``cpu`` ``small`` ``huge`` 等队列)和国产ARM节点(用于 ``arm128c256g`` 队列)的处理器指令集是不兼容的，需使用对应的镜像构建节点。

.. tip:: 请选择与目标主机(x86或arm)相匹配的容器构建节点。

从登录节点跳转X86容器构建节点::

.. console:: bash
   $ ssh build@container-x86
   $ hostname
   container-x86.pi.sjtu.edu.cn

从登录节点跳转ARM容器构建节点::

.. console:: bash
   $ ssh build@container-arm
   $ hostname
   container-arm.pi.sjtu.edu.cn

.. tip:: 出于安全考虑， ``container-x86`` 和 ``container-arm`` 节点每天 **23:59** 重启和清空节点数据，请及时转移容器构建节点上的数据。``build`` 为共享用户，请勿修改自己工作目录外的数据，以免影响其他用户的使用。

由于所有用户共享使用 ``build`` 用户，需要创建专属工作目录，在工作目录中构建镜像。
我们使用 ``mktemp -d`` 命令在 ``/tmp`` 目录下创建名字带有随机字符的工作目录。

.. console:: bash

   $ cd $(mktemp -d)
   $ pwd
   /tmp/tmp.sr7C5813M9

我们准备一个镜像定义文件 ``sample.def`` ，这个定义文件使用CentOS 8为基本镜像，安装编译器、OpenMPI等工具，编译OpenFOAM 8，内容如下::

    Bootstrap: docker
    From: centos:8
    
    %help
        This recipe provides an OpenFOAM-8 environment installed 
        with GCC and OpenMPI-4.
    
    %labels
        Author Fatih Ertinaz
    
    %post
        ### Install prerequisites
        yum groupinstall -y 'Development Tools'
        yum install -y wget git openssl-devel libuuid-devel
    
        ### Install OpenMPI
        # Why openmpi-4.x is needed: https://github.com/hpcng/singularity/issues/2590
        vrs=4.0.3
        wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${vrs}.tar.gz
        tar xf openmpi-${vrs}.tar.gz && rm -f openmpi-${vrs}.tar.gz
        cd openmpi-${vrs}
        ./configure --prefix=/opt/openmpi-${vrs}
        make all install
        make all clean
    
        ### Update environment - OpenMPI
        export MPI_DIR=/opt/openmpi-${vrs}
        export MPI_BIN=$MPI_DIR/bin
        export MPI_LIB=$MPI_DIR/lib
        export MPI_INC=$MPI_DIR/include
    
        export PATH=$MPI_BIN:$PATH
        export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH
    
        ### OpenFOAM version
        pkg=OpenFOAM
        vrs=8
    
        ### Install under /opt
        base=/opt/$pkg
        mkdir -p $base && cd $base
    
        ### Download OF
        wget -O - http://spack.pi.sjtu.edu.cn/mirror/openfoam-org/openfoam-org-8.0.tar.gz | tar xz
        mv $pkg-$vrs-version-$vrs $pkg-$vrs
    
        ## Download ThirdParty
        wget -O - http://spack.pi.sjtu.edu.cn/mirror/openfoam-org/ThirdParty-8.tar.gz | tar xz
        mv ThirdParty-$vrs-version-$vrs ThirdParty-$vrs
    
        ### Change dir to OpenFOAM-version
        cd $pkg-$vrs
        
        ### Get rid of unalias otherwise singularity fails
        sed -i 's,FOAM_INST_DIR=$HOME\/$WM_PROJECT,FOAM_INST_DIR='"$base"',g' etc/bashrc
        sed -i 's/alias wmUnset/#alias wmUnset/' etc/config.sh/aliases
        sed -i '77s/else/#else/' etc/config.sh/aliases
        sed -i 's/unalias wmRefresh/#unalias wmRefresh/' etc/config.sh/aliases
    
        ### Compile and install
        . etc/bashrc 
        ./Allwmake -j$(nproc) 2>&1 | tee log.Allwmake
    
        ### Clean-up environment
        rm -rf platforms/$WM_OPTIONS/applications
        rm -rf platforms/$WM_OPTIONS/src
    
        cd $base/ThirdParty-$vrs
        rm -rf build
        rm -rf gcc-* gmp-* mpfr-* binutils-* boost* ParaView-* qt-*
    
        strip $FOAM_APPBIN/*
    
        ### Source bashrc at runtime
        echo '. /opt/OpenFOAM/OpenFOAM-8/etc/bashrc' >> $SINGULARITY_ENVIRONMENT
    
    %environment
        export MPI_DIR=/opt/openmpi-4.0.3
        export MPI_BIN=$MPI_DIR/bin
        export MPI_LIB=$MPI_DIR/lib
        export MPI_INC=$MPI_DIR/include
    
        export PATH=$MPI_BIN:$PATH
        export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH
    
    %test
        . /opt/OpenFOAM/OpenFOAM-8/etc/bashrc
        icoFoam -help
    
    %runscript
        echo
        echo "OpenFOAM installation is available under $WM_PROJECT_DIR"
        echo

调用“容器化的Singularity”构建镜像，由于指令集的差异，使用的镜像标签也有x86和arm分别。

.. tip:: 在 ``container-x86`` 上请使用 ``sjtuhpc/centos7-singularity:x86`` ，在 ``container-arm`` 上请使用 ``sjtuhpc/centos7-singularity:x86`` 。

在 ``container-x86`` 节点上上构建镜像，构建的镜像保存在当前目录 ``sample-x86.sif`` ::

.. code:: bash

    $ docker run --privileged --rm -v \
         ${PWD}:/home/singularity \
         sjtuhpc/centos7-singularity:x86 \
         singularity build /home/singularity/sample-x86.sif /home/singularity/sample.def

在 ``container-arm`` 节点上上构建镜像，构建的镜像保存在当前目录 ``sample.sif`` ::

.. code:: bash

    $ docker run --privileged --rm -v \
         ${PWD}:/home/singularity \
         sjtuhpc/centos7-singularity:arm \
         singularity build /home/singularity/sample-arm.sif /home/singularity/sample.def

在镜像构建过程中“模拟”了root特权，因此生成镜像文文件属主是root::

.. code:: bash

    $  ls -alh *.sif
    -rwxr-xr-x 1 root root 475M Jun  3 22:43 sample-x86.sif

将构建出的镜像从 ``container`` 节点传回登录节点的家目录中::

.. code:: bash

   $ scp sample-x86.sif YOUR_USERNAME@login1:~/

然后编写作业脚本提交到作业调度系统。
下面这个作业脚本示例使用刚才构建的OpenFOAM镜像，完成了网格划分、模型求解、后处理等操作::

    #!/bin/bash
    
    #SBATCH --job-name=openfoam
    #SBATCH --partition=cpu
    #SBATCH -n 40
    #SBATCH --ntasks-per-node=40
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    
    ulimit -s unlimited
    ulimit -l unlimited
    
    module load openmpi/4.0.3-gcc-9.3.0
    
    export IMAGE_NAME=./8-centos8.sif
    
    singularity exec $IMAGE_NAME surfaceFeatures
    singularity exec $IMAGE_NAME blockMesh
    singularity exec $IMAGE_NAME decomposePar -copyZero
    mpirun -n $SLURM_NTASKS singularity exec $IMAGE_NAME snappyHexMesh -overwrite -parallel
    mpirun -n $SLURM_NTASKS singularity exec $IMAGE_NAME potentialFoam -parallel
    mpirun -n $SLURM_NTASKS singularity exec $IMAGE_NAME simpleFoam -parallel

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

参考资料
--------

- Singularity Quick Start https://sylabs.io/guides/3.4/user-guide/quick_start.html
- Docker Hub https://hub.docker.com/
- NVIDIA GPU CLOUD https://ngc.nvidia.com/
- Fakeroot feature of Singularity https://sylabs.io/guides/3.5/user-guide/fakeroot.html
