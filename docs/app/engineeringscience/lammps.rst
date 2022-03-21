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
| 2020   | |cpu|   | 容器     | 直接使用镜像                                |
+--------+---------+----------+---------------------------------------------+
| 2019   | |arm|   | 容器     | lammps/bisheng-1.3.3-lammps-2019            |
+--------+---------+----------+---------------------------------------------+

算例内容如下： `in.lj.txt` 
------------------------

.. code:: bash

   # 3d Lennard-Jones melt

   variable     x index 4
   variable     y index 4
   variable     z index 4
   
   variable     xx equal 20*$x
   variable     yy equal 20*$y
   variable     zz equal 20*$z
   
   units                lj
   atom_style   atomic
   
   lattice              fcc 0.8442
   region               box block 0 ${xx} 0 ${yy} 0 ${zz}
   create_box   1 box
   create_atoms 1 box
   mass         1 1.0
   
   velocity     all create 1.44 87287 loop geom
   
   pair_style   lj/cut 2.5
   pair_coeff   1 1 1.0 1.0 2.5
   
   neighbor     0.3 bin
   neigh_modify delay 0 every 20 check no
   
   fix          1 all nve
   
   run          10000


集群上的 LAMMPS
---------------

- `思源一号 LAMMPS`_

- `π2.0 LAMMPS`_

- `ARM LAMMPS`_


.. _思源一号 LAMMPS:

一. 思源一号 LAMMPS
---------------------

1. Intel编译器编译的版本
~~~~~~~~~~~~~~~~~~~~~~~~~~

脚本如下所示

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
      
   module load lammps/20210310-intel-2021.4.0-omp
   
   mpirun lmp -pk intel 0 omp 1 -sf intel -i in.lj

.. _π2.0 LAMMPS:

二. π2.0 LAMMPS
----------------

1. Intel编译器部署的版本
~~~~~~~~~~~~~~~~~~~~~~~~~~

调用镜像封装lammps(Intel CPU加速版本）示例脚本（intel_lammps.slurm）

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=intel_test
   #SBATCH --partition=cpu
   #SBATCH -N 2
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

2. CPU 版本自行编译
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

.. _ARM LAMMPS:

三. ARM LAMMPS
---------------

1. ARM版lammps(bisheng编译器+hypermpi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

脚本如下(lammps.slurm):

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lammps       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=96
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load lammps/bisheng-1.3.3-lammps-2019
   mpirun -x OMP_NUM_THREADS=1 lmp_aarch64_arm_hypermpi -in in.lj.txt

.. code:: bash

   $ sbatch lammps.slurm

运行结果(单位为：s,越低越好)
---------------------------------------

思源一号
~~~~~~~~

+------------------------------------------------+
|     lammps/20210310-intel-2021.4.0-omp         |
+=============+==========+===========+===========+
| 核数        | 64       | 128       | 192       |
+-------------+----------+-----------+-----------+
| Wall time   | 0:01:26  | 0:00:46   | 0:00:36   |
+-------------+----------+-----------+-----------+

π2.0
~~~~~

+-----------------------------------------------+
|                intel加速版本                  |          
+=============+==========+===========+==========+
| 核数        | 40       | 80        | 120      |
+-------------+----------+-----------+----------+
| Wall time   | 0:02:37  | 0:01:16   | 0:00:52  |
+-------------+----------+-----------+----------+

ARM
~~~

+------------------------------------+
| lammps/bisheng-1.3.3-lammps-2019   |
+==============+==========+==========+
| 核数         | 64       | 96       |
+--------------+----------+----------+
|  Wall time   | 0:07:26  | 0:04:43  |
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
