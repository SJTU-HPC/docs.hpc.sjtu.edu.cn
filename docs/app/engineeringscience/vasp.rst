.. _vasp:

VASP
====

简介
----

VASP 全称 Vienna Ab-initio Simulation Package，是维也纳大学 Hafner 小组开发的进行电子结构计算和量子力学-分子动力学模拟软件包。它是目前材料模拟和计算物质科学研究中最流行的商用软件之一。

VASP 使用需要得到 VASP 官方授权。请自行购买 VASP license 许可，下载和安装。如需协助安装或使用，请发邮件联系我们，附上课题组拥有 VASP license 的证明。

本文档将介绍如何使用集群上已部署到 VASP 5.4.4 和 6.2.1，以及如何自行编译 VASP

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

   module purge
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

   module purge

   module use /lustre/share/singularity/commercial-app
   module load vasp/5.4.4-intel

   ulimit -s unlimited
   ulimit -l unlimited

   export OMP_NUM_THREADS=2

   srun --mpi=pmi2 vasp_std

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
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive

   module purge
   module load openmpi/4.0.3-gcc-9.2.0
   mpirun singularity exec /lustre/share/singularity/commercial-app/vasp/5.4.4-arm.sif vasp_std

自行编译 VASP
-------------------

VASP 在集群上使用 intel 套件自行编译十分容易。下面以思源一号为例，介绍安装和使用方法。

1. 先申请计算节点，然后加载 intel 套件

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash       # 申请计算节点

   module purge
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

   module purge
   module load intel-oneapi-compilers/2021.4.0
   module load intel-oneapi-mpi/2021.4.0
   module load intel-oneapi-mkl/2021.4.0

   ulimit -s unlimited

   mpirun ~/vasp_std


参考资料
--------

-  `VASP 官网 <https://www.vasp.at/>`__