OpenFOAM
========

OpenFOAM（英文Open Source Field Operation and Manipulation的缩写，意为开源的场运算和处理软件）是对连续介质力学问题进行数值计算的C++自由软件工具包，其代码遵守GNU通用公共许可证。它可进行数据预处理、后处理和自定义求解器，常用于计算流体力学(CFD)领域。该软件由OpenFOAM基金会维护。

可用OpenFOAM版本
----------------

+------+-------+----------+--------------------------------------------------------------------+
| 版本 | 平台  | 构建方式 | 模块名                                                             |
+======+=======+==========+====================================================================+
| 7    | |cpu| | Spack    | openfoam-org/7-gcc-7.4.0-openmpi                                   |
+------+-------+----------+--------------------------------------------------------------------+
| 8    | |cpu| | 容器     | /lustre/share/img/x86/openfoam/8-gcc8-openmpi4-centos8.sif         |
+------+-------+----------+--------------------------------------------------------------------+
| 1712 | |cpu| | Spack    | openfoam/1712-gcc-7.4.0-openmpi                                    |
+------+-------+----------+--------------------------------------------------------------------+
| 1912 | |cpu| | Spack    | openfoam/1912-gcc-7.4.0-openmpi                                    |
+------+-------+----------+--------------------------------------------------------------------+
| 2012 | |cpu| | 容器     | /lustre/share/img/x86/openfoam/2012-gcc4-openmpi4-centos7.sif      |
+------+-------+----------+--------------------------------------------------------------------+
| 2106 | |cpu| | 容器     | /lustre/share/img/x86/openfoam/2106-gcc4-openmpi4-centos7.sif      |
+------+-------+----------+--------------------------------------------------------------------+
| 8    | |arm| | 容器     | /lustre/share/img/aarch64/openfoam/8-gcc8-openmpi4-centos8.sif     |
+------+-------+----------+--------------------------------------------------------------------+
| 1912 | |arm| | Spack    | openfoam/1912-gcc-9.3.0-openmpi                                    |
+------+-------+----------+--------------------------------------------------------------------+
| 2012 | |arm| | 容器     | /lustre/share/img/aarch64/openfoam/2012-gcc4-openmpi4-centos7.sif  |
+------+-------+----------+--------------------------------------------------------------------+
| 2106 | |arm| | 容器     | /lustre/share/img/aarch64/openfoam/2106-gcc4-openmpi4-centos7.sif  |
+------+-------+----------+--------------------------------------------------------------------+
| 1912 | |arm| | 容器     | /lustre/share/img/aarch64/openfoam/1912.sif                        |
+------+-------+----------+--------------------------------------------------------------------+



OpenFOAM基本使用
--------------------------------




思源一号上的openfoam-org7(Spack构建)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 从openfoam-org7的安装目录中将tutorials目录整个复制到自己目录下openfoamTest1目录中：

.. code:: bash
   
   module load openfoam-org/7-gcc-11.2.0-openmpi
   mkdir openfoamTest1
   cd openfoamTest1
   cp -rv $FOAM_TUTORIALS  ./

2. 为了运行motorBike算例(多核并行)，执行以下命令进入相对应目录：

.. code:: bash

   cd ./tutorials/incompressible//simpleFoam/motorBike


3. 在此目录下编写以下openfoam.slurm脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam      # 作业名
   #SBATCH --partition=64c512g      # 64c512g队列
   #SBATCH --ntasks-per-node=6      # 每节点核数
   #SBATCH -n 6                     # 作业核心数
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited
   
   module load openmpi/4.1.1-gcc-11.2.0
  
   cp $FOAM_TUTORIALS/resources/geometry/motorBike.obj.gz constant/triSurface/ 
   surfaceFeatures 
   blockMesh 
   decomposePar -copyZero 
   mpirun -np 6 snappyHexMesh -overwrite -parallel 
   mpirun -np 6 patchSummary -parallel 
   mpirun -np 6 potentialFoam -parallel 
   mpirun -np 6 simpleFoam -parallel 
   reconstructParMesh -constant 
   reconstructPar -latestTime

.. caution::

   某些情况下，当并行核数达到一定数目时，直接运行大规模并行作业可能会出现报错，这是超算集群上各节点之间的连接方式造成的.这时需要修改一下MPI的默认执行方式，比如说将 mpirun -np 6 simpleFoam -parallel 改为 mpirun -np 6 -mca btl self,vader,tcp simpleFoam -parallel.


4. 使用 ``sbatch`` 提交作业：

.. code:: bash

   sbatch openfoam.slurm

5. 运行结束后即可在该目录下看到如下结果：

.. code:: bash

    0
    500
    9953216.err
    9953216.out
    Allclean
    Allrun
    constant
    postProcessing
    processor0
    processor1
    processor2
    processor3
    processor4
    processor5
    openfoam.slurm
    system



思源一号上的openfoam2106(Spack构建)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 从openfoam2106的安装目录中将tutorials目录整个复制到自己目录下openfoamTest1目录中：

.. code:: bash
   
   module load openfoam/2106-gcc-8.3.1-openmpi
   mkdir openfoamTest1
   cd openfoamTest1
   cp -rv $FOAM_TUTORIALS  ./

2. 为了运行motorBike算例(多核并行)，执行以下命令进入相对应目录：

.. code:: bash

   cd ./tutorials/incompressible//simpleFoam/motorBike


3. 在此目录下编写以下openfoam.slurm脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam      # 作业名
   #SBATCH --partition=64c512g      # 64c512g队列
   #SBATCH --ntasks-per-node=6      # 每节点核数
   #SBATCH -n 6                     # 作业核心数
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited
   
   module load openmpi/4.1.1-gcc-8.3.1
   
   ./Allclean
   ./Allrun

4. 使用 ``sbatch`` 提交作业：

.. code:: bash

   sbatch openfoam.slurm

5. 运行结束后即可在该目录下看到如下结果：

.. code:: bash

 0.orig
 500
 Allclean
 Allrun
 constant
 log.blockMesh
 log.checkMesh
 log.decomposePar
 log.patchSummary
 log.potentialFoam
 log.reconstructPar
 log.reconstructParMesh
 log.simpleFoam
 log.snappyHexMesh
 log.surfaceFeatureExtract
 log.topoSet
 openfoam.slurm
 postProcessing
 processor0
 processor1
 processor2
 processor3
 processor4
 processor5
 system


pi2.0上的openfoam-org7(Spack构建)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. 从openfoam-org7的安装目录中将tutorials目录整个复制到自己目录下openfoamTest1目录中：

.. code:: bash

   module load openfoam-org/7-gcc-7.4.0-openmpi
   mkdir openfoamTest1
   cd openfoamTest1
   cp -rv $FOAM_TUTORIALS  ./
   
     

2. 运行cavity算例(单核串行)，执行以下命令进入相对应目录：

.. code:: bash

   cd ./tutorials/incompressible/icoFoam/cavity/cavity

3. 此时可以看到以下0、constant、system三个目录(一个典型的openfoam算例均包含这三个目录)：

.. code:: bash

  ├── 0
  │   ├── p
  │   └── U
  ├── constant
  │   └── transportProperties
  └── system
    ├── blockMeshDict
    ├── controlDict
    ├── fvSchemes
    └── fvSolution


*其中 0目录主要包含待求解问题的边界条件和初始条件；
constant目录主要包含物性参数、湍流模型参数、更高级的物理模型等；
system目录主要包含计算时间和数值求解格式等计算参数。
这三个目录包含了待求解问题所必须指定的所有物理参数和计算参数，用户可根据自己的需求进行合理修改以提高计算结果的准确性。*

4. 在此目录下编写以下openfoam.slurm脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam       # 作业名
   #SBATCH --partition=small         # small队列
   #SBATCH --ntasks-per-node=1       # 每节点核数
   #SBATCH -n 1                      # 作业核心数
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   #module load openfoam-org/7-gcc-7.4.0-openmpi

   blockMesh
   icoFoam

5. 使用 ``sbatch`` 提交作业：

.. code:: bash

   sbatch openfoam.slurm

6. 运行结束后会看到constant目录下多出了一个polyMesh目录，该目录保存了计算用的网格信息；而同级目录下多出了0.1、0.2、0.3、0.4、0.5这五个目录，这几个目录记录了在五个不同时刻的物理场的计算结果：

.. code:: bash

  ├── 0
  │   ├── p
  │   └── U
  ├── 0.1
  │   ├── p
  │   ├── phi
  │   ├── U
  │   └── uniform
  │       └── time
  ├── 0.2
  │   ├── p
  │   ├── phi
  │   ├── U
  │   └── uniform
  │       └── time
  ├── 0.3
  │   ├── p
  │   ├── phi
  │   ├── U
  │   └── uniform
  │       └── time
  ├── 0.4
  │   ├── p
  │   ├── phi
  │   ├── U
  │   └── uniform
  │       └── time
  ├── 0.5
  │   ├── p
  │   ├── phi
  │   ├── U
  │   └── uniform
  │       └── time
  ├── constant
  │   ├── polyMesh
  │   │   ├── boundary
  │   │   ├── faces
  │   │   ├── neighbour
  │   │   ├── owner
  │   │   └── points
  │   └── transportProperties
  ├── openfoam.slurm
  └── system
    ├── blockMeshDict
    ├── controlDict
    ├── fvSchemes
    └── fvSolution


自行构建OpenFOAM镜像
------------------------------------

以OpenFOAM-org7为例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 从pi2.0登录节点跳转至容器构建节点(只能从pi2.0跳转，思源一号不行)：

.. code:: bash

    ssh build@container-x86

2. 创建和进入临时工作目录：

.. code:: bash

    cd $(mktemp -d)

  
3. 创建如下镜像定义文件openfoam7-gcc4-openmpi4-centos7.def，可按需修改：

.. code:: bash

   Bootstrap: docker
   From: centos:7

   %help
      This recipe provides an OpenFOAM-7 environment installed 
      with GCC 4 and OpenMPI-4 on CentOS 7.

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
      vrs=7

      ### Install under /opt
      base=/opt/$pkg
      mkdir -p $base && cd $base

      ### Download OF
      wget -O - http://spack.pi.sjtu.edu.cn/mirror/openfoam-org/openfoam-org-7.0.tar.gz | tar xz
      mv $pkg-$vrs-version-$vrs $pkg-$vrs

      ## Download ThirdParty
      wget -O - http://spack.pi.sjtu.edu.cn/mirror/openfoam-org/ThirdParty-7.tar.gz | tar xz
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
      echo '. /opt/OpenFOAM/OpenFOAM-7/etc/bashrc' >> $SINGULARITY_ENVIRONMENT

   %environment
      export MPI_DIR=/opt/openmpi-4.0.3
      export MPI_BIN=$MPI_DIR/bin
      export MPI_LIB=$MPI_DIR/lib
      export MPI_INC=$MPI_DIR/include

      export PATH=$MPI_BIN:$PATH
      export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH

   %test
      . /opt/OpenFOAM/OpenFOAM-7/etc/bashrc
      icoFoam -help

   %runscript
      echo
      echo "OpenFOAM installation is available under $WM_PROJECT_DIR"
      echo
   
4. 执行以下命令构建镜像，大约会消耗2-3小时：

.. code:: bash

     docker run --privileged --rm -v \
     ${PWD}:/home/singularity \
     sjtuhpc/centos7-singularity:x86 \
     singularity build /home/singularity/openfoam7-gcc4-openmpi4-centos7.sif /home/singularity/openfoam7-gcc4-openmpi4-centos7.def

5. 将构建出的容器镜像传回自己家目录的根目录(用户需根据自身具体情况将hpcpzz@login2进行修改，和自己对应起来)：

.. code:: bash

   scp ./openfoam7-gcc4-openmpi4-centos7.sif hpcpzz@login2:~/

6. 回到自己的家目录，测试镜像是否能够正常使用：

.. code:: bash

   singularity exec ./openfoam7-gcc4-openmpi4-centos7.sif blockMesh

7. 得到以下结果则表示镜像构建成功：

.. code:: bash

   /*---------------------------------------------------------------------------*\
    =========                 |
    \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
     \\    /   O peration     | Website:  https://openfoam.org
       \\  /    A nd           | Version:  7
        \\/     M anipulation  |
   \*---------------------------------------------------------------------------*/
   Build  : 7
   Exec   : /opt/OpenFOAM/OpenFOAM-7/platforms/linux64GccDPInt32Opt/bin/blockMesh
   Date   : Nov 25 2022
   Time   : 13:30:41
   Host   : "cas332.pi.sjtu.edu.cn"
   PID    : 174880
   I/O    : uncollated
   Case   : /lustre/home/acct-hpc/hpcpzz
   nProcs : 1
   fileModificationChecking : Monitoring run-time modified files using timeStampMaster (fileModificationSkew 10)
   allowSystemOperations : Allowing user-supplied system call operations

   // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
   Create time



   --> FOAM FATAL ERROR: 
   cannot find file "/lustre/home/acct-hpc/hpcpzz/system/controlDict"

      From function virtual Foam::autoPtr<Foam::ISstream> Foam::fileOperations::uncollatedFileOperation::readStream(Foam::regIOobject&, const Foam::fileName&, const Foam::word&, bool) const
      in file global/fileOperations/uncollatedFileOperation/uncollatedFileOperation.C at line 538.

   FOAM exiting



在自己的目录下自行源码编译OpenFOAM
----------------------------------------------------------------------------------------

以OpenFOAM-org10为例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 从登录节点申请计算资源：

.. code:: bash

   srun -p cpu -N 1 --ntasks-per-node=40  --pty /bin/bash(思源一号)
   或者
   srun -p 64c512g -n 10 --pty /bin/bash(pi2.0)

2. 加载编译所需模块：

.. code:: bash

   module load gcc
   module load openmpi

3. 执行以下命令源码编译openfoam10 (其中10为版本号，用户可根据自身需求改为7，8，9)：

.. code:: bash

   cd $HOME 
   mkdir OpenFOAM 
   cd OpenFOAM 
   git clone https://github.com/OpenFOAM/OpenFOAM-10.git 
   git clone https://github.com/OpenFOAM/ThirdParty-10.git 
   source OpenFOAM-10/etc/bashrc 
   cd OpenFOAM-10 
   ./Allwmake -j 
   sed -i '$a source $HOME/OpenFOAM/OpenFOAM-10/etc/bashrc' $HOME/.bashrc

4. 测试是否编译成功:

.. code:: bash

   cd $HOME/OpenFOAM/OpenFOAM-10/tutorials/multiphase/interFoam/RAS/DTCHull
   ./Allrun





参考资料
--------

-  `Openfoam-org 官网 <https://openfoam.org/>`__
-  `Openfoam-org github 地址 <https://github.com/OpenFOAM?tab=repositories/>`__



