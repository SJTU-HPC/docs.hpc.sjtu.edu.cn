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
3. 按需定制Singularity镜像。

.. TODO: 使用Singularity加载集群预编译的镜像
.. TODO: ===================================
.. TODO: 胡筱婧

使用Singularity拉取远端镜像
===========================

您可以使用 ``singularity pull`` 拉取远端预编译的镜像，从而直接使用外部预预编译镜像仓库提供的丰富软件资源。
Singularity可以从Docker Hub(以 ``docker://`` 开头)、Singularity Hub(以 ``shub://`` 开头)等地址拉取镜像。
如下命令从Docker Hub拉取CentOS 8标准镜像，保存成名为 ``cento8.sif`` 的镜像文件：

.. code:: console

    $ singularity pull centos8.sif docker://centos:centos8

查看生成的镜像文件

.. code:: console

    $ ls centos8.sif
    centos8.sif

加载容器镜像，并且在容器环境中运行 ``cat`` 程序，查看容器内 ``/etc/redhat-release`` 文件的内容，然后在宿主环境中运行同样命令，对比结果：
  
.. code:: console

    $ singularity exec centos.sif cat /etc/redhat-release
    CentOS Linux release 8.3.2011
    $ cat /etc/redhat-release
    CentOS Linux release 7.7.1908 (Core)

运行结果显示，我们成功在CentOS 7操作系统上加载了一个CentOS 8容器镜像。

.. tip:: Singularity镜像文件(Singularity Image File, sif)是一种内容只读的文件格式，其文件内容不能被修改。

按需定制Singularity镜像
=======================

Singularity使用“镜像定义文件”(Definition File)描述镜像构建过程。
镜像定义文是一个文本文件，描述了构建镜像使用的基本镜像、构建过程执行的命令，其中不少命令通常需要root特权，如软件包管理命令 ``yum``, ``apt-get`` 等等。
运行Singularity容器构建命令 ``singularity build`` ，以及构建过程中在镜像内部的系统相关命令，都需要root特权，因此构建容器的操作通常在您自管的、具有root特权的计算机上完成。
在HPC等共享集群环境中，普通用户没有root权限，因而无法在集群上定制和构建镜像。


在π超算集群上，我们采用“容器化的Singularity”，允许用户在一个受限的环境内以普通用户身份“模拟”root特权，编写自定义的镜像文件，从定义文件构建镜像，并将镜像传回集群使用。

.. tip:: 构建Singularity容器镜像通常需要root特权，通常超算集群不支持这样的操作。π超算集群的“容器化的Singularity”允许用户编写、构建和传回自定义容器镜像。

首先从登录节点使用用户名 ``build`` 跳转到专门用于构建容器镜像的节点。
需要注意的是，X86节点(用于 ``cpu`` ``small`` ``huge`` 等队列)和国产ARM节点(用于 ``arm128c256g`` 队列)的处理器指令集是不兼容的，需使用对应的镜像构建节点。

.. tip:: 请选择与目标主机(x86或arm)相匹配的容器构建节点。

从登录节点跳转X86容器构建节点：

.. code:: console

   $ ssh build@container-x86
   $ hostname
   container-x86.pi.sjtu.edu.cn

从登录节点跳转ARM容器构建节点：

.. code:: console

   $ ssh build@container-arm
   $ hostname
   container-arm.pi.sjtu.edu.cn

.. caution:: 出于安全考虑， ``container-x86`` 和 ``container-arm`` 节点每天 **23:59** 重启节点并清空数据，请及时转移容器构建节点上的数据。``build`` 为共享用户，请勿修改自己工作目录外的数据，以免影响其他用户的使用。

由于所有用户共享使用 ``build`` 用户，需要创建专属工作目录，在工作目录中构建镜像。
我们使用 ``mktemp -d`` 命令在 ``/tmp`` 目录下创建名字带有随机字符的工作目录。

.. code:: console

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

在 ``container-x86`` 节点上上构建镜像，构建的镜像保存在当前目录 ``sample-x86.sif`` ：

.. code:: console

    $ docker run --privileged --rm -v \
         ${PWD}:/home/singularity \
         sjtuhpc/centos7-singularity:x86 \
         singularity build /home/singularity/sample-x86.sif /home/singularity/sample.def

在 ``container-arm`` 节点上上构建镜像，构建的镜像保存在当前目录 ``sample.sif`` ：

.. code:: console

    $ docker run --privileged --rm -v \
         ${PWD}:/home/singularity \
         sjtuhpc/centos7-singularity:arm \
         singularity build /home/singularity/sample-arm.sif /home/singularity/sample.def

在镜像构建过程中“模拟”了root特权，因此生成镜像文文件属主是root：

.. code:: console

    $  ls -alh *.sif
    -rwxr-xr-x 1 root root 475M Jun  3 22:43 sample-x86.sif

将构建出的镜像从 ``container`` 节点传回登录节点的家目录中：

.. code:: console

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

参考资料
========

- Singularity Quick Start https://sylabs.io/guides/3.4/user-guide/quick_start.html
- Docker Hub https://hub.docker.com/
- NVIDIA GPU CLOUD https://ngc.nvidia.com/
