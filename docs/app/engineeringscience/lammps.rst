.. _lammps:

LAMMPS
======

简介
----

LAMMPS 是大规模原子分子并行计算代码，在原子、分子及介观体系计算中均有重要应用。

可用的版本
----------

+--------+---------+----------+---------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                      |
+========+=========+==========+=============================================+
| 2021   | |cpu|   | spack    | lammps/20210310-intel-2021.4.0-omp 思源一号 |
+--------+---------+----------+---------------------------------------------+
| 2020   | |cpu|   | 容器     | lammps/2020-cpu                             |
+--------+---------+----------+---------------------------------------------+
| 2020   | |cpu|   | 容器     | 直接使用镜像                                |
+--------+---------+----------+---------------------------------------------+
| 2020   | |gpu|   | 容器     | lammps/2020-dgx-kokkos                      |
+--------+---------+----------+---------------------------------------------+
| 2019   | |arm|   | 容器     | lammps/bisheng-1.3.3-lammps-2019            |
+--------+---------+----------+---------------------------------------------+

算例下载
---------

.. code:: bash

   mkdir ~/lammps && cd ~/lammps
   wget https://lammps.sandia.gov/inputs/in.lj.txt

`in.lj.txt` 文件的最后一行步数设置为 `40000`

.. code:: bash

   run          40000

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

module load lammps/20210310-intel-2021.4.0-omp

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
      
   module load lammps/20210310-intel-2021.4.0-omp
   
   mpirun lmp -pk intel 0 omp 1 -sf intel -i in.lj.txt
  
module load lammps/20210310-intel-2021.4.0-omp

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module purge
   module load oneapi
   module load lammps/20210310-intel-2021.4.0

   ulimit -s unlimited
   ulimit -l unlimited
   export OMP_NUM_THREADS=1
   mpirun lmp -i in.lj.txt

2. π2.0上的Slurm 脚本
~~~~~~~~~~~~~~~~~~~~~~

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

   mpirun singularity run  $IMAGE_PATH  lmp -pk intel 0 omp 1 -sf intel -i ${INPUT_FILE} 
   
用下方语句提交作业:

.. code:: bash
      
   sbatch intel_lammps.slurm


4. CPU 版本自行编译
~~~~~~~~~~~~~~~~~~~

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的
Lammps. 下面以在 π 集群为例介绍 lammps 的自行安装

a) 从官网下载 lammps，推荐安装最新的稳定版：

.. code:: bash

   $ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz

b) 由于登录节点禁止运行作业和并行编译，请申请计算节点资源用来编译
   lammps，并在编译结束后退出：

.. code:: bash

   $ srun -p small -n 8 --pty /bin/bash

c) 加载 Intel 模块：

.. code:: bash

   $ module load intel-parallel-studio/cluster.2020.1

d) 编译 (以额外安装 MANYBODY 和 Intel 加速包为例)

.. code:: bash

   $ tar xvf lammps-stable.tar.gz
   $ cd lammps-XXXXXX
   $ cd src
   $ make                           #查看编译选项
   $ make package                   #查看包
   $ make yes-intel                 #"make yes-"后面接需要安装的 package 名字
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
   module load intel-parallel-studio/cluster.2020.1

   ulimit -s unlimited
   ulimit -l unlimited

   srun --mpi=pmi2 ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i in.lj.txt


.. _GPU版本 LAMMPS:

二. GPU版本
-----------

1. GPU 版本 LAMMPS + kokkos
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

1. ARM版lammps(bisheng编译器+hypermpi)
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

运行结果(单位为：timesteps/s,越高越好)
---------------------------------------

思源一号
~~~~~~~~

+------------------------------------------------+
|     lammps/20210310-intel-2021.4.0-omp         |
+=============+==========+===========+===========+
| 核数        | 64       | 128       | 256       |
+-------------+----------+-----------+-----------+
| Performance | 7890.438 | 10366.877 | 12598.343 |
+-------------+----------+-----------+-----------+

π2.0
~~~~~

+----------------------------------------------+
|              lammps/2020-cpu                 |
+=============+==========+==========+==========+
| 核数        | 40       | 80       | 160      |
+-------------+----------+----------+----------+
| Performance | 1861.775 | 3023.928 | 5057.443 |
+-------------+----------+----------+----------+

+-----------------------------------------------+
|                intel加速版本                  |          
+=============+==========+===========+==========+
| 核数        | 40       | 80        | 160      |
+-------------+----------+-----------+----------+
| Performance | 4391.882 | 6403.898  | 9131.615 |
+-------------+----------+-----------+----------+

AI集群
~~~~~~

+----------------------------------------------+
|            lammps/2020-dgx-kokkos            |
+=============+==========+==========+==========+
| 核数:GPU    | 6:1      | 12:2     | 18:3     |
+-------------+----------+----------+----------+
| Performance | 4212.983 | 1139.140 | 1166.863 |
+-------------+----------+----------+----------+

ARM
~~~

+------------------------------------+
| lammps/bisheng-1.3.3-lammps-2019   |
+==============+==========+==========+
| 核数         | 64       | 96       |
+--------------+----------+----------+
|  Performance | 2010.122 | 2776.084 |
+--------------+----------+----------+

建议
~~~~

通过分析上述结果，我们推荐您使用如下两个版本提交作业。

.. code:: bash

   module load lammps/20210310-intel-2021.4.0-omp               思源一号   
   /lustre/share/singularity/modules/lammps/20-user-intel.sif   π2.0

参考资料
--------

-  `LAMMPS 官网 <https://lammps.sandia.gov/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
