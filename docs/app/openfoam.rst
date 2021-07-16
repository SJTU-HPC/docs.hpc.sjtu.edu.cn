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
| 1712 | |cpu| | Spack    | openfoam/1712-gcc-7.4.0-openmpi                                    |
+------+-------+----------+--------------------------------------------------------------------+
| 1912 | |cpu| | Spack    | openfoam/1912-gcc-7.4.0-openmpi                                    |
+------+-------+----------+--------------------------------------------------------------------+
| 2106 | |cpu| | 容器     | /lustre/share/img/x86/openfoam/2106-gcc4-openmpi4-centos7.sif      |
+------+-------+----------+--------------------------------------------------------------------+
| 1912 | |arm| | Spack    | openfoam/1912-gcc-9.3.0-openmpi                                    |
+------+-------+----------+--------------------------------------------------------------------+
| 8    | |arm| | 容器     | /lustre/share/img/aarch64/openfoam/8-gcc8-openmpi4-centos8.sif     |
+------+-------+----------+--------------------------------------------------------------------+

提交OpenFOAM作业
----------------

### CPU版OpenFoam(使用Spack预编译版本)

准备作业脚本 ``openfoam.slurm`` ，内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam       # 作业名
   #SBATCH --partition=cpu           # cpu队列
   #SBATCH --ntasks-per-node=40      # 每节点核数
   #SBATCH -n 80                     # 作业核心数80(两个节点)
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load openfoam/1912-gcc-7.4.0-openmpi

   srun --mpi=pmi2 icoFoam -parallel

使用 ``sbatch`` 提交作业：

.. code:: bash

   $ sbatch openfoam.slurm

### CPU版OpenFoam(使用容器)

准备作业脚本 ``openfoam.slurm`` ，内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam       # 作业名
   #SBATCH --partition=cpu           # cpu队列
   #SBATCH --ntasks-per-node=40      # 每节点核数
   #SBATCH -n 80                     # 作业核心数80(两个节点)
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load openmpi/3.1.5-gcc-4.8.5

   ulimit -s unlimited
   ulimit -l unlimited

   export IMAGE_NAME=/lustre/share/img/x86/openfoam/2106-gcc4-openmpi4-centos7.sif

   singularity exec $IMAGE_NAME blockMesh
   mpirun -n $SLURM_NTASKS singularity exec $IMAGE_NAME simpleFoam -parallel

使用 ``sbatch`` 提交作业：

.. code:: bash

   $ sbatch openfoam.slurm

### ARM版OpenFoam(使用Spack预编译版本)

准备作业脚本 ``openfoam.slurm`` ，内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam          # 作业名
   #SBATCH --partition=arm128c256g      # arm128c256g队列                
   #SBATCH --ntasks-per-node=128        # 每节点核数
   #SBATCH -n 256                       # 作业核心数256(两个节点)
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load openfoam/1912-gcc-9.3.0-openmpi

   srun --mpi=pmi2 icoFoam -parallel

在ARM登录节点使用 ``sbatch`` 提交作业：

.. code:: bash

   $ sbatch openfoam.slurm


### ARM版OpenFoam(使用容器)

准备作业脚本 ``openfoam.slurm`` ，内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openfoam          # 作业名
   #SBATCH --partition=arm128c256g      # arm128c256g队列                
   #SBATCH --ntasks-per-node=128        # 每节点核数
   #SBATCH -n 256                       # 作业核心数256(两个节点)
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load openmpi/4.0.3-gcc-9.3.0

   ulimit -s unlimited
   ulimit -l unlimited

   export IMAGE_NAME=/lustre/share/img/x86/openfoam/8-gcc8-openmpi4-centos8.sif

   singularity exec $IMAGE_NAME blockMesh
   mpirun -n $SLURM_NTASKS singularity exec $IMAGE_NAME simpleFoam -parallel

使用 ``sbatch`` 提交作业：

.. code:: bash

   $ sbatch openfoam.slurm

编译OpenFOAM
------------

如果您需要从源代码构建OpenFOAM，我们强烈建议您使用超算平台提供的非特权容器构建方法(:ref:`dockerized_singularity`)，以确保编译过程能顺利完成。

### 编译适用于CPU平台的OpenFOAM(构建容器)

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

### 编译适用于ARM平台的OpenFOAM(构建容器)

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

参考链接
--------

- Openfoam官方网站 https://openfoam.org/
- OpenFOAM中文维基页面  
- Singularity文档 https://sylabs.io/guides/
