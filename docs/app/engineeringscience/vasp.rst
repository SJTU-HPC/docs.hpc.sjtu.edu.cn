.. _vasp:

VASP
====

简介
----

VASP 全称 Vienna Ab-initio Simulation Package，是维也纳大学 Hafner 小组开发的进行电子结构计算和量子力学-分子动力学模拟软件包。它是目前材料模拟和计算物质科学研究中最流行的商用软件之一。

VASP 使用需要得到 VASP 官方授权。请自行购买 VASP license 许可，下载和安装。如需协助安装或使用，请发邮件联系我们，附上课题组拥有 VASP license 的证明。

本文档将介绍如何使用集群上已部署的 VASP 5.4.4 和 6.2.1，以及如何自行编译 VASP。

集群上的 VASP
---------------

- `思源一号 VASP`_

- `π2.0 VASP`_

- `ARM VASP`_


.. _思源一号 VASP:

思源一号 VASP
~~~~~~~~~~~~~~~~~~~~~~~~

若已拥有 VASP license，请邮件联系我们，提供课题组拥有 VASP license 的证明，并注明是 VASP5 还是 VASP6，我们将添加该 VASP module 的使用权限

经测试，思源一号使用默认的 ``OMP_NUM_THREADS=1`` 速度比其它设置更好，故无需额外设置该参数

下面 slurm 脚本以 vasp 6.2.1 为例。若使用 vasp 5.4.4，请将 ``module load`` 那行换成 ``module load vasp/5.4.4-intel-2021.4.0``

.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp
   #SBATCH -p 64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load vasp/6.2.1-intel-2021.4.0-cuda-11.5.0

   ulimit -s unlimited

   mpirun vasp_std

.. _π2.0 VASP:

π2.0 VASP
~~~~~~~~~~~~~~~~~~~~~~~~

若已拥有 VASP license，请邮件联系我们，提供课题组拥有 VASP license 的证明，并注明是 VASP5 还是 VASP6，我们将添加该 VASP module 的使用权限

请注意，π2.0 上推荐使用 ``OMP_NUM_THREADS=2`` ，速度较默认的 ``OMP_NUM_THREADS=1`` 提升近 20%

slurm 里，若使用 CPU 节点，须确保 ``OMP_NUM_THREADS * ntasks-per-node = 总核数`` 。例如：

* 1 个 CPU 节点， ``OMP_NUM_THREADS=2`` ， ``ntasks-per-node=20``
  
* 2 个 CPU 节点， ``OMP_NUM_THREADS=2`` ， ``ntasks-per-node=40``

下面 slurm 脚本以 vasp 5.4.4 为例：

.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp
   #SBATCH -p cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=20
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module use /lustre/share/singularity/commercial-app
   module load vasp/5.4.4-intel

   ulimit -s unlimited
   ulimit -l unlimited

   export OMP_NUM_THREADS=2

   srun --mpi=pmi2 vasp_std

另外，模块 ``vasp/5.4.4-intel-2021.4.0`` 的使用方法如下所示

.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp
   #SBATCH -p cpu
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load vasp/5.4.4-intel-2021.4.0

   ulimit -s unlimited

   mpirun vasp_std

.. _ARM VASP:

ARM VASP
~~~~~~~~~~~~~~~~~~~~~~~~

若已拥有 VASP license，请邮件联系我们，提供课题组拥有 VASP license 的证明，并注明是 VASP5 还是 VASP6，我们将添加该 VASP module 的使用权限  

.. code:: bash

   #!/bin/bash

   #SBATCH -J vasp_arm
   #SBATCH -p arm128c256g
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=128

   module load openmpi/4.0.3-gcc-9.2.0
   mpirun singularity exec /lustre/share/singularity/commercial-app/vasp/5.4.4-arm.sif vasp_std

自行编译 VASP
-------------------

VASP 在集群上使用 intel 套件自行编译十分容易。下面以思源一号为例，介绍CPU版本的安装和使用方法。

1. 先申请计算节点，然后加载 intel 套件

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash       # 申请计算节点

   module load intel-oneapi-compilers/2021.4.0
   module load intel-oneapi-mpi/2021.4.0
   module load intel-oneapi-mkl/2021.4.0

2. 解压缩 VASP 安装包，进入 ``vasp.x.x.x`` 文件夹（可看到 ``arch``, ``src`` 等文件夹）

.. code:: bash

   cp arch/makefile.include.linux_intel makefile.include

3. 输入 ``make`` 开始编译，预计十分钟左右完成

.. code:: bash

   make

请注意，为了避免编译出错，推荐直接使用 make，不要添加 -jN (若一定要使用，请使用完整的命令： ``make DEPS=1 -jN`` )

编译完成后，bin 文件夹里将出现三个绿色的文件： ``vasp_std``, ``vasp_gam``, ``vasp_ncl``

可将 ``vasp_std`` 复制到 ``home/bin`` 里，后续可以直接调用：

.. code:: bash

   mkdir ~/bin       # 若 home 下未曾建过 bin，则新建一个；若已有，请略过此句
   cp vasp_std ~/bin

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

   module load intel-oneapi-compilers/2021.4.0
   module load intel-oneapi-mpi/2021.4.0
   module load intel-oneapi-mkl/2021.4.0

   ulimit -s unlimited

   mpirun ~/vasp_std

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

   mkdir ~/bin       # 若 home 下未曾建过 bin，则新建一个；若已有，请略过此句
   cp vasp_std ~/bin

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

VASP 算例及测试
---------------------

以 64 原子的 Si AIMD 熔化为例，各使用 40 核，思源一号与 π 2.0 的测试结果：

===================== ===================== =====================
      setting             思源一号 40核          π 2.0 40核
OMP_NUM_THREADS       CPU time used (sec)   CPU time used (sec)
===================== ===================== =====================
1                     19                    94
2                     23                    31
4                     39                    39
===================== ===================== =====================

测试结果说明：

* 思源一号推荐使用 ``OMP_NUM_THREADS=1``
  
* π 2.0 推荐使用 ``OMP_NUM_THREADS=2``

* 思源一号 VASP 计算速度明显优于 π 2.0

本示例相关说明：

1. VASP 运行需要最基本的 ``INCAR``, ``POSCAR``, ``POTCAR``, ``KPOINTS`` 四个文件。全部文件已放置于思源一号共享文件夹：

.. code:: bash

   /dssg/share/sample/vasp

2. VASP 算例运行方法：
      
.. code:: bash

   cp -r /dssg/share/sample/vasp ~
   cd vasp
   sbatch slurm.sub

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
