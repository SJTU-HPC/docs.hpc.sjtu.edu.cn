.. _lammps:

LAMMPS
======

简介
----

LAMMPS是一个大规模经典分子动力学代码，用于大规模原子/分子的并行模拟。LAMMPS在软材料（生物分子、聚合物）、固态材料（金属、半导体）和粗颗粒或介观系统方面具有重要作用。可用来模拟原子，或者更一般地说，作为原子、介观或连续尺度上的并行粒子模拟器。

可用的版本
----------

+--------+---------+----------+-----------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                  |
+========+=========+==========+=========================================+
| 2021   | |cpu|   | spack    | lammps/20210310-intel-2021.4.0 思源一号 |
+--------+---------+----------+-----------------------------------------+
| 2020   | |cpu|   | 容器     | lammps/2020-cpu                         |
+--------+---------+----------+-----------------------------------------+
| 2020   | |cpu|   | 容器     | 直接使用镜像                            |
+--------+---------+----------+-----------------------------------------+
| 2020   | |gpu|   | 容器     | lammps/2020-dgx                         |
+--------+---------+----------+-----------------------------------------+
| 2020   | |gpu|   | 容器     | lammps/2020-dgx-kokkos                  |
+--------+---------+----------+-----------------------------------------+
| 2019   | |arm|   | 容器     | lammps/bisheng-1.3.3-lammps-2019        |
+--------+---------+----------+-----------------------------------------+
| 2021   | |arm|   | spack    | 20210310-gcc-9.3.0-openblas-openmpi     |
+--------+---------+----------+-----------------------------------------+


算例下载
---------

.. code:: bash

   mkdir ~/lammps && cd ~/lammps
   wget https://lammps.sandia.gov/inputs/in.lj.txt

集群上的 LAMMPS
---------------

- `CPU版本 LAMMPS`_

- `GPU版本 LAMMPS`_

- `ARM版本 LAMMPS`_


.. _CPU版本 LAMMPS:

一. CPU 版本
-------------

1. 思源一号上的调用脚本
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
      
   module purge
   module load oneapi
   module load lammps/20210310-intel-2021.4.0
   
   ulimit -s unlimited
   ulimit -l unlimited
   
   mpirun lmp -i in.lj.txt
   
运行结果如下所示

.. code:: bash

   Step Temp E_pair E_mol TotEng Press 
          0         1.44   -6.7733681            0   -4.6134356   -5.0197073 
      40000   0.69567179   -5.6686654            0   -4.6251903   0.73582061 
   Loop time of 6.25411 on 128 procs for 40000 steps with 32000 atoms
   
   Performance: 2762981.774 tau/day, 6395.791 timesteps/s
   100.0% CPU use with 128 MPI tasks x 1 OpenMP threads
   
2. π2.0上的Slurm 脚本
~~~~~~~~~~~~~~~~~~~~~

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
   
   srun --mpi=pmi2 lmp -i in.lj.txt

用下方语句提交作业

.. code:: bash

   sbatch slurm.test

运行结果如下所示

.. code:: bash

   Step Temp E_pair E_mol TotEng Press 
          0         1.44   -6.7733681            0   -4.6134356   -5.0197073 
      40000   0.69605629   -5.6690032            0   -4.6249514    0.7424604 
   Loop time of 13.3113 on 80 procs for 40000 steps with 32000 atoms
   
   Performance: 1298148.809 tau/day, 3004.974 timesteps/s
   99.7% CPU use with 80 MPI tasks x 1 OpenMP threads

3. Intel加速版
~~~~~~~~~~~~~~~

调用镜像封装lammps(Intel CPU加速版本）示例脚本（intel_lammps.slurm）

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
   export INPUT_FILE=in.lj.txt
   export IMAGE_PATH=/lustre/share/singularity/modules/lammps/20-user-intel.sif
   KMP_BLOCKTIME=0 mpirun singularity run  $IMAGE_PATH  lmp -pk intel 0 omp 1 -sf intel -i ${INPUT_FILE} 
   
用下方语句提交作业:

.. code:: bash
   
   sbatch intel_lammps.slurm


4. CPU 版本自行编译
~~~~~~~~~~~~~~~~~~~

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的
Lammps.

a) 从官网下载 lammps，推荐安装最新的稳定版：

.. code:: bash

   $ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz

b) 由于登录节点禁止运行作业和并行编译，请申请计算节点资源用来编译
   lammps，并在编译结束后退出：

.. code:: bash

   $ srun -p small -n 8 --pty /bin/bash

c) 加载 Intel 模块：

.. code:: bash

   $ module load intel-parallel-studio/cluster.2019.5

d) 编译 (以额外安装 MANYBODY 和 USER-MEAMC 包为例)

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

e) 测试脚本

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
   module load intel-parallel-studio/cluster.2019.5

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   ulimit -s unlimited
   ulimit -l unlimited

   srun ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i in.lj.txt


.. _GPU版本 LAMMPS:

二. GPU版本
-----------

1. GPU版本脚本
~~~~~~~~~~~~~~

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

   srun --mpi=pmi2 lmp -in in.lj.txt

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

2. GPU 版本 LAMMPS + kokkos
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

   srun --mpi=pmi2 lmp -k on g 2 t 12  -sf kk -pk kokkos comm device -in in.lj.txt

其中，g 2 t 12 意思是使用 2 张 GPU 和 12 个线程。-sf kk -pk kokkos comm
device 是 LAMMPS 的 kokkos 设置，可以用这些默认值

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

.. _ARM版本 LAMMPS:

三. ARM版本
-----------

1. ARM脚本
~~~~~~~~~~

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

   mpirun -n $SLURM_NTASKS lmp -in in.lj.txt

在 `ARM 节点 <../login/index.html#arm>`__\ 上使用如下指令提交（若在 π2.0 登录节点上提交将出错）：

.. code:: bash

   $ sbatch lammps.slurm

2. ARM版lammps(bisheng编译器+hypermpi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

脚本如下(lammps.slurm):

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lammps       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=16
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load lammps/bisheng-1.3.3-lammps-2019
   mpirun -x OMP_NUM_THREADS=1 lmp_aarch64_arm_hypermpi -in in.lj.txt

.. code:: bash

   $ sbatch lammps.slurm

运行结果
--------

思源一号
~~~~~~~~

+-----------------------------------------+
|     lammps/20210310-intel-2021.4.0      |
+===========+=========+=========+=========+
| 核数      | 64      | 128     | 192     |
+-----------+---------+---------+---------+
| Loop time | 10.6259 | 6.25411 | 5.56981 |
+-----------+---------+---------+---------+

π2.0
~~~~

+-----------------------------------------+
|            lammps/2020-cpu              |
+===========+=========+=========+=========+
| 核数      | 40      | 80      | 120     |
+-----------+---------+---------+---------+
| Loop time | 21.8741 | 13.3113 | 10.2851 |
+-----------+---------+---------+---------+

+-----------------------------------------+
|              intel加速版本              |          
+===========+=========+=========+=========+
| 核数      | 40      | 80      | 120     |
+-----------+---------+---------+---------+
| Loop time | 9.10169 | 6.2462  | 5.68533 |
+-----------+---------+---------+---------+

AI集群
~~~~~~

+-----------------------------------------+
|          lammps/2020-dgx-kokkos         |
+===========+=========+=========+=========+
| 核数:GPU  | 6:1     | 12:2    | 18:3    |
+-----------+---------+---------+---------+
| Loop time | 9.49446 | 35.1142 | 34.2799 |
+-----------+---------+---------+---------+

ARM
~~~

+----------------------------------+
| lammps/bisheng-1.3.3-lammps-2019 |
+==============+=========+=========+
| 核数         | 64      | 96      |
+--------------+---------+---------+
| Loop time    | 19.8993 | 14.4088 |
+--------------+---------+---------+

建议
~~~~

通过分析上述结果，速度最快的版本为思源一号和π2.0部署的intel加速版,我们推荐您使用这两个版本。

.. code:: bash

   module load lammps/20210310-intel-2021.4.0
   /lustre/share/singularity/modules/lammps/20-user-intel.sif

参考资料
--------

-  `LAMMPS 官网 <https://lammps.sandia.gov/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
