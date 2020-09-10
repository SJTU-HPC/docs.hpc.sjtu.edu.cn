LAMMPS
======

简介
----

LAMMPS is a large scale classical molecular dynamics code, and stands
for Large-scale Atomic/Molecular Massively Parallel Simulator. LAMMPS
has potentials for soft materials (biomolecules, polymers), solid-state
materials (metals, semiconductors) and coarse-grained or mesoscopic
systems. It can be used to model atoms or, more generically, as a
parallel particle simulator at the atomic, meso, or continuum scale.

Pi上的LAMMPS
------------

Pi 上有多种版本的 LAMMPS:

-  |cpu| `cpu <#cpu-lammps>`__
-  |gpu| `gpu <#gpu-lammps>`__
-  |arm| `arm <#arm-lammps>`__

使用CPU版本LAMMPS
-----------------

|cpu| (CPU) LAMMPS 模块调用
~~~~~~~~~~~~~~~~~~~~~~~~~~~

查看 Pi 上已编译的软件模块:

.. code:: bash

   $ module avail lammps

调用该模块:

.. code:: bash

   $ module load lammps/20190807-intel-19.0.5-impi

|cpu| (CPU) LAMMPS 的 Slurm 脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J lammps_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
   module load lammps/20190807-intel-19.0.5-impi

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   ulimit -s unlimited
   ulimit -l unlimited

   srun lmp -i YOUR_INPUT_FILE

|cpu| (CPU) LAMMPS 提交作业
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   $ sbatch slurm.test

|cpu| (CPU) LAMMPS 自行编译
~~~~~~~~~~~~~~~~~~~~~~~~~~~

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的
Lammps.

1. 从官网下载 lammps，推荐安装最新的稳定版：

.. code:: bash

   $ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz

2. 由于登陆节点禁止运行作业和并行编译，请申请计算节点资源用来编译
   lammps，并在编译结束后退出：

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

3. 加载 Intel-mpi 模块：

.. code:: bash

   $ module purge
   $ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

4. 编译 (以额外安装 USER-MEAMC 包为例)

.. code:: bash

   $ tar xvf lammps-stable.tar.gz
   $ cd lammps-XXXXXX
   $ cd src
   $ make                           #查看编译选项
   $ make package                   #查看包
   $ make yes-user-meamc            #"make yes-"后面接需要安装的 package 名字
   $ make -j 4 intel_cpu_intelmpi   #开始编译

5. 测试脚本

编译成功后，将在 src 文件夹下生成 lmp_intel_cpu_intelmpi.
后续调用，请给该文件的路径，比如
``~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi``

.. code:: bash

   #!/bin/bash

   #SBATCH -J lammps_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   ulimit -s unlimited
   ulimit -l unlimited

   srun ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i YOUR_INPUT_FILE

|gpu| 使用 GPU 版本的 LAMMPS
----------------------------

Pi 集群已预置 NVIDIA GPU CLOUD 提供的优化镜像，调用该镜像即可运行
LAMMPS，无需单独安装，目前版本为 2019.8。该容器文件位于
/lustre/share/img/lammps_7Aug2019.simg

以下 slurm 脚本，在 dgx2 队列上使用 1 块 gpu，并配比 6 cpu 核心，调用
singularity 容器中的 GROMACS：

.. code:: bash

   #!/bin/bash
   #SBATCH -J gromacs_gpu_test
   #SBATCH -p dgx2
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 6
   #SBATCH --ntasks-per-node=6
   #SBATCH --gres=gpu:1
   #SBATCH -N 1

   IMAGE_PATH=/lustre/share/img/lammps_7Aug2019.simg

   ulimit -s unlimited
   ulimit -l unlimited

   singularity run $IMAGE_PATH -i YOUR_INPUT_FILE

使用如下指令提交：

.. code:: bash

   $ sbatch lammps_gpu.slurm

参考资料
--------

-  LAMMPS官网 https://lammps.sandia.gov/
-  NVIDIA GPU CLOUD https://ngc.nvidia.com
-  Singularity文档 https://sylabs.io/guides/3.5/user-guide/

.. include:: /defs.hrst
