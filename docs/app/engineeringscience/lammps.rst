.. _lammps:

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

π 集群上的 LAMMPS
--------------------

π 集群上有多种版本的 LAMMPS:

- `CPU版本 LAMMPS`_

- `GPU版本 LAMMPS`_

- `ARM版本 LAMMPS`_

.. _CPU版本 LAMMPS:


CPU 版本
~~~~~~~~

查看 π 集群 上已编译的软件模块:

.. code:: bash

   module av lammps

推荐使用 lammps/2020-cpu，经测试，该版本在 π 集群上运行速度最好，且安装有丰富的 LAMMPS package：

ASPHERE BODY CLASS2 COLLOID COMPRESS CORESHELL DIPOLE GRANULAR KSPACE
MANYBODY MC MISC MLIAP MOLECULE OPT PERI POEMS PYTHON QEQ REPLICA RIGID
SHOCK SNAP SPIN SRD VORONOI USER-BOCS USER-CGDNA USER-CGSDK USER-COLVARS
USER-DIFFRACTION USER-DPD USER-DRUDE USER-EFF USER-FEP USER-MEAMC
USER-MESODPD USER-MISC USER-MOFFF USER-OMP USER-PHONON USER-REACTION
USER-REAXC USER-SDPD USER-SPH USER-SMD USER-UEF USER-YAFF

调用该模块:

.. code:: bash

   module load lammps/2020-cpu

CPU 版本 Slurm 脚本
~~~~~~~~~~~~~~~~~~~

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=cpu
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40

   module purge
   module load lammps/2020-cpu

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 lmp -i YOUR_INPUT_FILE

用下方语句提交作业

.. code:: bash

   sbatch slurm.test

Intel加速CPU版本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

调用镜像封装lammps(Intel CPU加速版本）示例脚本（intel_lammps.slurm）:

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=intel_test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module purge
   module load oneapi/2021
   export INPUT_FILE=in.eam
   export IMAGE_PATH=/lustre/share/singularity/modules/lammps/20-user-intel.sif
   KMP_BLOCKTIME=0 mpirun -n 40 singularity run  $IMAGE_PATH  lmp -pk intel 0 omp 1 -sf intel -i ${INPUT_FILE} 


用下方语句提交作业:

.. code:: bash
   
   sbatch intel_lammps.slurm


（进阶）CPU 版本自行编译
~~~~~~~~~~~~~~~~~~~~~~~~

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的
Lammps.

1. 从官网下载 lammps，推荐安装最新的稳定版：

.. code:: bash

   $ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz

2. 由于登录节点禁止运行作业和并行编译，请申请计算节点资源用来编译
   lammps，并在编译结束后退出：

.. code:: bash

   $ srun -p small -n 8 --pty /bin/bash

3. 加载 Intel 模块：

.. code:: bash

   $ module load intel-parallel-studio/cluster.2019.4-intel-19.0.4

4. 编译 (以额外安装 MANYBODY 和 USER-MEAMC 包为例)

.. code:: bash

   $ tar xvf lammps-stable.tar.gz
   $ cd lammps-XXXXXX
   $ cd src
   $ make                           #查看编译选项
   $ make package                   #查看包
   $ make yes-user-meamc            #"make yes-"后面接需要安装的 package 名字
   $ make yes-manybody
   $ make ps                        #查看计划安装的包列表 
   $ make -j 8 intel_cpu_intelmpi   #开始编译

5. 测试脚本

编译成功后，将在 src 文件夹下生成 lmp_intel_cpu_intelmpi.
后续调用，请给该文件的路径，比如
``~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi``\ 。脚本名称可设为
slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH -J lammps_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load intel-parallel-studio/cluster.2019.4-intel-19.0.4

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   ulimit -s unlimited
   ulimit -l unlimited

   srun ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i YOUR_INPUT_FILE


.. _GPU版本 LAMMPS:

GPU版本
~~~~~~~

GPU 版本速度跟 intel CPU 版本基本相同

π 集群 上提供了 GPU 版本的 LAMMPS 2020。经测试，LJ 和 EAM 两 Benchmark
算例与同等计算费用的 CPU 基本一样。建议感兴趣的用户针对自己的算例，测试
CPU 和 GPU 计算效率，然后决定使用哪一种平台。

以下 slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu 核心，使用
GPU 版 LAMMPS。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lmp_test
   #SBATCH --partition=dgx2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=12
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:2

   ulimit -s unlimited
   ulimit -l unlimited

   module load lammps/2020-dgx

   srun --mpi=pmi2 lmp -in in.eam

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

GPU 版本 LAMMPS + kokkos
------------------------

GPU 版本速度跟 intel CPU 版本基本相同

π 集群上提供了 GPU + kokkos 版本的 LAMMPS 15Jun2020。采用容器技术，使用
LAMMPS 官方提供给 NVIDIA 的镜像，针对 Tesla V100 的 GPU
做过优化，性能很好。经测试，LJ 和 EAM 两 Benchmark 算例与同等计算费用的
CPU 基本一样。建议感兴趣的用户针对自己的算例，测试 CPU 和 GPU
计算效率，然后决定使用哪一种平台。

以下 slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu 核心，使用
GPU kokkos 版的 LAMMPS。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lmp_test
   #SBATCH --partition=dgx2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=6
   #SBATCH --gres=gpu:2

   ulimit -s unlimited
   ulimit -l unlimited

   module load lammps/2020-dgx-kokkos

   srun --mpi=pmi2 lmp -k on g 2 t 12  -sf kk -pk kokkos comm device -in in.eam

其中，g 2 t 12 意思是使用 2 张 GPU 和 12 个线程。-sf kk -pk kokkos comm
device 是 LAMMPS 的 kokkos 设置，可以用这些默认值

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

.. _ARM版本 LAMMPS:

ARM版本
~~~~~~~

脚本如下(lammps.slurm):

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lmp_test
   #SBATCH --partition=arm128c256g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 256
   #SBATCH --ntasks-per-node=128

   ulimit -s unlimited
   ulimit -l unlimited

   module purge
   module load openmpi/4.0.3-gcc-9.3.0
   module load lammps/20210310-gcc-9.3.0-openblas-openmpi

   mpirun -n $SLURM_NTASKS lmp -in in.eam

在 `ARM 节点 <../login/index.html#arm>`__\ 上使用如下指令提交（若在 π2.0 登录节点上提交将出错）：

.. code:: bash

   $ sbatch lammps.slurm

参考资料
--------

-  `LAMMPS 官网 <https://lammps.sandia.gov/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__

