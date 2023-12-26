.. _croco:

Croco 
================

简介
-------
CROCO 是一个基于 ROMS_AGRIF 构建的海洋建模系统，由 IRD、INRIA、CNRS、IFREMER 和 SHOM（法国致力于环境科学和应用数学的机构）维护。 CROCO 的一个重要目标是解决非常精细的尺度（特别是在沿海地区）及其与更大尺度的相互作用。它包括非静水力求解器、海浪-大气耦合、不断演变的沉积物动力学和海洋生物地球化学、用于平流和混合的新高阶数值方案以及专用 I/O 服务器 (XIOS) 等新功能。源代码附带了一个用于预处理和后处理的工具箱 CROCO_TOOLS。 

自行安装croco步骤，以在Pi2.0 KOS系统中安装为例
------------------------------------------------

下载croco与croco_tools源代码并解压

.. code:: console
    
    $ wget https://data-croco.ifremer.fr/CODE_ARCHIVE/croco-v1.3.tar.gz
    $ wget https://data-croco.ifremer.fr/CODE_ARCHIVE/croco_tools-v1.3.tar.gz
    $ tar zxvf croco-v1.3.tar.gz
    $ mv croco-v1.3 croco
    $ tar croco_tools-v1.3.tar.gz
    $ mv croco_tools-v1.3 croco_tools


新建编译文件夹Run

.. code:: console

    $ cd croco
    $ ./create_config.bash

修改编译文件cppdefs.h，启用并行计算，根据用户需求设置配置类型，例如修改为Basin

.. code:: console

    $ cd Run
    $ vim cppdefs.h 
    $ # /* Parallelization */选项：修改`# undef  MPI` 为 `# define MPI`
    $ # /* Configuration Name */选项：修改`# define BENGUELA_LR` 为 `# undef BENGUELA_LR # define Basin`


修改编译文件param.h，设置并行计算调用的核数，例如修改为512核。

.. code:: console

    $ vim param.h 
    $ # 修改 `parameter (NP_XI=1,  NP_ETA=4,  NNODES=NP_XI*NP_ETA)` 为 `parameter (NP_XI=16,  NP_ETA=32,  NNODES=NP_XI*NP_ETA)`

注意：该软件提交计算调用的核数需要与param.h设置的核数一致，核数np=NNODES=NP_XI*NP_ETA

修改编译文件jobcomp

.. code:: console

    $ vim jobcomp  # 修改如下内容：
    $ FC=$FC
    $ MPIF90=$MPIF90
    $ MPIDIR=$(dirname $(dirname $(which $MPIF90) ))
    $ MPILIB="-L$MPIDIR/lib -lmpi -limf -lm"
    $ MPIINC="-I$MPIDIR/include"

申请计算节点，调用编译软件，声明编译器

.. code:: console

    $ srun -p cpu -n 8 --pty /bin/bash
    $ module load oneapi/2021.4.0
    $ export CC=icc
    $ export FC=ifort
    $ export F90=ifort
    $ export F77=ifort
    $ export MPIF90=mpiifort

开始编译

.. code:: console

    $ ./jobcomp > jobcomp.log

编译完成后生成可执行文件croco以及包含源文件的文件夹Compile

提交任务脚本，以Pi2.0集群为例
--------------------------------

.. code:: console

    #!/bin/bash

    #SBATCH --job-name=test       
    #SBATCH --partition=cpu       
    #SBATCH -N 13                
    #SBATCH --ntasks-per-node=40 
    #SABTCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    ulimit -s unlimited
    ulimit -l unlimited
    ulimit -n 4096

    module load oneapi/2021.4.0
    module load netcdf-fortran/4.5.2-intel-2021.4.0

    mpirun -np 512 ./croco croco.in #提交计算的核数需要与param.h编译文件设置的核数一致！

参考资料
-----------

-  `croco <https://www.croco-ocean.org>`__
-  `croco documentation <https://croco-ocean.gitlabpages.inria.fr/croco_doc/index.html>`__