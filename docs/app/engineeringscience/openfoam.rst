OpenFOAM
========

OpenFOAM（英文Open Source Field Operation and Manipulation的缩写，意为开源的场运算和处理软件）是对连续介质力学问题进行数值计算的C++自由软件工具包，其代码遵守GNU通用公共许可证。它可进行数据预处理、后处理和自定义求解器，常用于计算流体力学(CFD)领域。该软件由OpenFOAM基金会维护。

可用OpenFOAM版本
----------------

+------+-------+----------+--------------------------------------------------------------------+
| 版本 | 平台  | 构建方式 | 模块名                                                             |
+======+=======+==========+====================================================================+
| 7    | cpu   | Spack    | openfoam-org/7-gcc-7.4.0-openmpi                                   |
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

以pi2.0上的OpenFoam7为例：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.从OpenFoam7的安装目录中将tutorials目录整个复制到自己(本人hpcpzz，用户根据自己的实际情况进行修改即可)的目录下openfoamTest目录中：

.. code:: bash
   
   mkdir openfoamTest
   cp -r /lustre/opt/cascadelake/linux-centos7-skylake_avx512/gcc-7.4.0/openfoam-org-7-hf7fehnzmia4muicuqvlcaki7y2iqx2x/tutorials   /lustre/home/acct-hpc/hpcpzz/openfoamTest  

2.为了运行cavity(方腔流动)算例，执行以下命令进入相对应目录：

.. code:: bash

   cd /lustre/home/acct-hpc/hpcpzz/openfoamTest/tutorials/incompressible/icoFoam/cavity/cavity

3.此时可以看到以下0、constant、system三个目录(一个典型的openfoam算例均包含这三个目录)：

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


0 目录主要包含待求解问题的边界条件和初始条件；
constant 目录主要包含物性参数、湍流模型参数、更高级的物理模型等；
system 目录主要包含计算时间和数值求解格式等计算参数。

这三个目录包含了待求解问题所必须指定的所有物理参数和计算参数，用户可根据自己的需求进行合理修改以提高计算结果的准确性。

4.在此目录下编写以下openfoam.slurm脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam       # 作业名
   #SBATCH --partition=small         # small队列
   #SBATCH --ntasks-per-node=4       # 每节点核数
   #SBATCH -n 4                      # 作业核心数4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load openfoam-org/7-gcc-7.4.0-openmpi

   blockMesh
   icoFoam

5.使用 ``sbatch`` 提交作业：

.. code:: bash

   sbatch openfoam.slurm

6.运行结束后会看到constant目录下多出了一个polyMesh目录，该目录保存了计算用的网格信息；而同级目录下多出了0.1、0.2、0.3、0.4、0.5这五个目录，这几个目录记录了在五个不同时刻的物理场的计算结果。

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



编译OpenFOAM
------------

如果您需要从源代码构建OpenFOAM，我们强烈建议您使用超算平台提供的非特权容器构建方法，以确保编译过程能顺利完成。

编译适用于CPU平台的OpenFOAM(构建容器)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从登录节点跳转至容器构建X86节点：

.. code:: bash

   # ssh build@container-x86

创建和进入临时工作目录：

.. code:: bash

   $ cd $(mktemp -d)
   $ pwd
   /tmp/tmp.sr7C5813M9
  
下载镜像定义文件，按需定制修改：

.. code:: bash

   $ wget https://raw.githubusercontent.com/SJTU-HPC/hpc-base-container/dev/base/openfoam/2012-gcc4-openmpi4-centos7.def
   
构建Singularity容器镜像，大约会消耗2-3小时：

.. code:: bash

   $ docker run --privileged --rm -v \
     ${PWD}:/home/singularity \
     sjtuhpc/centos7-singularity:x86 \
     singularity build /home/singularity/2012-gcc4-openmpi4-centos7.sif /home/singularity/2012-gcc4-openmpi4-centos7.def

将构建出的容器镜像传回家目录，参考上文的作业脚本(容器版)提交作业。

.. code:: bash

   $ scp 2012-gcc4-openmpi4-centos7.sif YOUR_USER_NAME@login1:~/

编译适用于ARM平台的OpenFOAM(构建容器)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从登录节点跳转至容器构建ARM节点：

.. code:: bash

   # ssh build@container-arm

创建和进入临时工作目录：

.. code:: bash

   $ cd $(mktemp -d)
   $ pwd
  
下载镜像定义文件，按需定制修改：

.. code:: bash

   $ wget https://raw.githubusercontent.com/SJTU-HPC/hpc-base-container/dev/base/openfoam/8-gcc8-openmpi4-centos8.def
   
构建Singularity容器镜像，大约会消耗2-3小时：

.. code:: bash

   $ docker run --privileged --rm -v \
     ${PWD}:/home/singularity \
     sjtuhpc/centos7-singularity:arm \
     singularity build /home/singularity/8-gcc8-openmpi4-centos8.def /home/singularity/8-gcc8-openmpi4-centos8.def

将构建出的容器镜像传回家目录，参考上文的作业脚本(容器版)提交作业。

.. code:: bash

   $ scp 8-gcc8-openmpi4-centos8.sif YOUR_USER_NAME@login1:~/

编译OpenFOAM6，添加相应的自定义功能模块，此处的镜像只包含OpenFOAM6编译所依赖的基础环境
----------------------------------------------------------------------------------------

.. code:: bash

   cd $HOME
   mkdir OpenFOAM
   cd OpenFOAM
   cp /lustre/opt/contribute/cascadelake/openfoam/img/OpenFOAM-6.tar.gz ./
   cp /lustre/opt/contribute/cascadelake/openfoam/img/ThirdParty-6.tar.gz ./
   tar xf OpenFOAM-6.tar.gz
   tar xf ThirdParty-6.tar.gz
   echo "alias of6='source \$HOME/OpenFOAM/OpenFOAM-6/etc/bashrc WM_LABEL_SIZE=64 FOAMY_HEX_MESH=yes'" >> ~/.bashrc
   singularity shell /lustre/opt/contribute/cascadelake/openfoam/img/openfoam6_base.sif
   ln -s /usr/bin/mpicc.openmpi OpenFOAM-6/bin/mpicc
   ln -s /usr/bin/mpirun.openmpi OpenFOAM-6/bin/mpirun
   source $HOME/OpenFOAM/OpenFOAM-6/etc/bashrc WM_LABEL_SIZE=64 FOAMY_HEX_MESH=yes
   source ~/.bashrc
   of6
   cd $WM_THIRD_PARTY_DIR
   export QT_SELECT=qt4
   ./makeParaView -python -mpi -python-lib /usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0 > log.makePV 2>&1
   wmRefresh
   cd $WM_PROJECT_DIR
   export QT_SELECT=qt4
   ./Allwmake -j 4 > log.make 2>&1
   ./Allwmake -j 4 > log.make 2>&1

编译成功时，输入icoFoam -help会显示如下信息

.. code:: bash

   Usage: icoFoam [OPTIONS]
   options:
     -case <dir>       specify alternate case directory, default is the cwd
     -noFunctionObjects
                       do not execute functionObjects
     -parallel         run in parallel
     -roots <(dir1 .. dirN)>
                       slave root directories for distributed running
     -srcDoc           display source code in browser
     -doc              display application documentation in browser
     -help             print the usage

每次重新进入OpenFOAM6环境中，输入如下命令，然后根据需要添加自定义功能模块

.. code:: bash

   singularity shell /lustre/opt/contribute/cascadelake/openfoam/img/openfoam6_base.sif
   of6

参考资料
--------

- Openfoam官方网站 https://openfoam.org/
- OpenFOAM中文维基页面  
- Singularity文档 https://sylabs.io/guides/
