.. _vasp:

VASP
====

简介
----

VASP 全称 Vienna Ab-initio Simulation Package，是维也纳大学 Hafner 小组开发的进行电子结构计算和量子力学-分子动力学模拟软件包。它是目前材料模拟和计算物质科学研究中最流行的商用软件之一。

.. attention::

   1. VASP 使用需要得到 VASP 官方授权，请自行购买 VASP license 许可，下载和安装。
   2. 若课题组已购买 VASP ，可发送邮件提供课题组的 VASP license 证明，并注明需要使用的集群名称以及集群上具体的VASP模块名称，我们将添加该 VASP module 的使用权限。
   3. 如果需要编译第三方插件或者需要修改源码版本的VASP，请提前参考VASP官网或者第三方插件官网确认需要安装的VASP版本是否支持。

本文档将介绍如何使用集群上已部署的 VASP ，以及如何自行编译 VASP。

集群上可用的VASP版本
-----------------------
+--------+---------+----------+----------+-----------------------------------------------------+
| 版本   | 平台    | 构建方式 | 集群     | 模块名                                              |
+========+=========+==========+==========+=====================================================+
| 5.4.4  | |cpu|   | 源码     | 思源一号 |vasp/5.4.4-intel-2021.4.0                            |
+--------+---------+----------+----------+-----------------------------------------------------+
| 6.2.1  | |gpu|   | 源码     | 思源一号 |vasp/6.2.1-intel-2021.4.0-cuda-11.5.0                |
+--------+---------+----------+----------+-----------------------------------------------------+
| 6.3.2  | |cpu|   | 源码     | 思源一号 |vasp/6.3.2-vtst-intel-2021.4.0                       |
+--------+---------+----------+----------+-----------------------------------------------------+
| 5.4.4  | |cpu|   | 源码     | Pi 2.0   |vasp/5.4.4-intel-2021.4.0                            |
+--------+---------+----------+----------+-----------------------------------------------------+
| 6.3.2  | |cpu|   | 源码     | Pi 2.0   |vasp/6.3.2-intel-2021.4.0                            |
+--------+---------+----------+----------+-----------------------------------------------------+

思源一号 VASP
~~~~~~~~~~~~~~~~~~~~~~~~

下面 slurm 脚本以 vasp 6.3.2 为例。若使用其他模块，比如 vasp 5.4.4，请将 ``module load`` 那行换成 ``module load vasp/5.4.4-intel-2021.4.0``

.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp
   #SBATCH -p 64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   
   module purge
   module load vasp/6.3.2-intel-2021.4.0

   ulimit -s unlimited
   ulimit -l unlimited

   mpirun -np $SLURM_NPROCS vasp_std


π2.0 VASP
~~~~~~~~~~~~~~~~~~~~~~~~

下面 slurm 脚本以 vasp 5.4.4 为例：

.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp
   #SBATCH -p cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load vasp/5.4.4-intel-2021.4.0

   ulimit -s unlimited
   ulimit -l unlimited

   mpirun -np $SLURM_NPROCS vasp_std

自行编译 VASP
-------------------

VASP 在集群上使用 intel 套件自行编译十分容易，下面介绍CPU版本的安装和使用方法。

1. 先申请计算节点，然后加载 intel 套件

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash       # 思源集群申请计算节点
   srun -p cpu -n 4 --pty /bin/bash           # Pi2.0集群申请计算节点

   module load oneapi/2021.4.0                # 加载intel套件

1. 解压缩 VASP 安装包，进入 ``vasp.x.x.x`` 文件夹（可看到 ``arch``, ``src`` 等文件夹）

.. code:: bash

   cp arch/makefile.include.linux_intel makefile.include

3. 输入 ``make`` 开始编译，预计十分钟左右完成

.. code:: bash

   make

请注意，为了避免编译出错，推荐直接使用 make，不要添加 -jN (若一定要使用，请使用完整的命令： ``make DEPS=1 -jN`` )

编译完成后，bin 文件夹里将出现三个绿色的文件： ``vasp_std``, ``vasp_gam``, ``vasp_ncl``

可将 ``vasp_std`` 复制到 ``$HOME/bin`` 里，后续可以直接调用：

.. code:: bash

   mkdir -p ~/bin       # 若 home 下未曾建过 bin，则新建一个；若已有，请略过此句
   cp bin/vasp_std ~/bin

4. 使用
   
.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp
   #SBATCH -p 64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   
   module purge
   module load oneapi/2021.4.0

   ulimit -s unlimited
   ulimit -l unlimited

   mpirun -np $SLURM_NPROCS ~/bin/vasp_std

VASP-GPU版本编译安装

由于VASP为商业软件，需要用户自行申请license、在官网自行下载源码包。
下面介绍如何在思源一号上的a100节点上，编译安装GPU版本VASP，本文以6.3.0为例编译nvhpc+acc版本，其他版本请参考vasp官网。

1. 先申请计算节点，然后加载编译环境

.. code:: bash

   srun -n 16 --gres=gpu:1 -p a100 --pty /bin/bash       # 申请计算节点

   module load nvhpc/23.3-gcc-11.2.0
   module load oneapi/2021.4.0
   module unload intel-oneapi-mpi/2021.4.0
   module load gcc/11.2.0 cuda/11.8.0 

2. 解压缩 VASP 安装包，进入 ``vasp.x.x.x`` 文件夹，可看到 ``arch``, ``src`` 等文件夹。

.. code:: bash

   cp arch/makefile.include.nvhpc_acc makefile.include
   ＃　修改makefile.include文件
   ＃　删除或注释 BLAS and LAPACK，scaLAPACK，FFTW，新增　MKL 设置
   ＃ Intel MKL (FFTW, BLAS, LAPACK, and scaLAPACK
   MKLROOT    ?= /dssg/opt/icelake/linux-centos8-icelake/gcc-8.5.0/intel-oneapi-mkl-2021.4.0-r7h6alnulyzgb6iqvxhovmwrajvwbqxf/mkl/2021.4.0/
   LLIBS      += -Mmkl -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64
   INCS       += -I$(MKLROOT)/include/fftw
   
3. 输入 ``make`` 开始编译

.. code:: bash

   make

请注意，为了避免编译出错，推荐直接使用 make，不要添加 -jN (若一定要使用，请使用完整的命令： ``make DEPS=1 -jN`` )

编译完成后，bin 文件夹里将出现三个绿色的文件： ``vasp_std``, ``vasp_gam``, ``vasp_ncl``

可将 ``vasp_std`` 复制到 ``home/bin`` 里，后续可以直接调用：

.. code:: bash

   mkdir -p ~/bin       # 若 home 下未曾建过 bin，则新建一个；若已有，请略过此句
   cp bin/vasp_std ~/bin

4. 作业脚本
   
.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp-gpu
   #SBATCH -p a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=16
   #SBATCH --gres=gpu:1

   module load nvhpc/23.3-gcc-11.2.0
   module load oneapi/2021.4.0
   module unload intel-oneapi-mpi/2021.4.0
   module load gcc/11.2.0 cuda/11.8.0
   ulimit -s unlimited
   ulimit -l unlimited

   mpirun -np 1 ~/bin/vasp_std

VASP 算例
---------------------

以 64 原子的 Si AIMD 熔化为例，本示例相关说明：

1. VASP 运行需要最基本的 ``INCAR``, ``POSCAR``, ``POTCAR``, ``KPOINTS`` 四个文件。全部文件已放置于思源一号共享文件夹：

.. code:: bash

   /dssg/share/sample/vasp

2. VASP 算例运行方法：
      
.. code:: bash

   cp -r /dssg/share/sample/vasp ~
   cd vasp
   sbatch run.slurm

3. 下面是该示例的 ``INCAR`` 文件内容：

.. code:: bash

   SYSTEM = cd Si

   ! ab initio
   ISMEAR = 0        ! Gaussian smearing
   SIGMA  = 0.1      ! smearing in eV

   LREAL  = Auto     ! projection operators in real space

   ALGO   = VeryFast ! RMM-DIIS for electronic relaxation
   PREC   = Low      ! precision
   ISYM   = 0        ! no symmetry imposed

   ! MD
   IBRION = 0        ! MD (treat ionic dgr of freedom)
   NSW    = 60       ! no of ionic steps
   POTIM  = 3.0      ! MD time step in fs

   MDALGO = 2        ! Nosé-Hoover thermostat
   SMASS  = 1.0      ! Nosé mass

   TEBEG  = 2000     ! temperature at beginning
   TEEND  = 2000     ! temperature at end
   ISIF   = 2        ! update positions; cell shape and volume fixed

   NCORE = 4




参考资料
--------

-  `VASP 官网 <https://www.vasp.at/>`__
