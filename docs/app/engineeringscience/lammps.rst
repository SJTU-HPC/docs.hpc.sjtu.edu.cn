.. _lammps:

LAMMPS
======

简介
----

LAMMPS 是大规模原子分子并行计算代码，在原子、分子及介观体系计算中均有重要应用，并行效率高，广泛应用于材料、物理、化学等模拟。

专题培训
--------
-  `“交我算” lammps软件编译与使用 <https://jbox.sjtu.edu.cn/l/u1bEaN>`__

可用的版本
----------

+----------+-------+-----------+------------------------------------+
| 集群     | 平台  |版本       | 模块名                             |
+==========+=======+===========+====================================+
| 思源一号 | |cpu| | 20230328  | lammps/20230328-intel-2021.4.0-omp |
+----------+-------+-----------+------------------------------------+
| 思源一号 | |cpu| | 20220324  | lammps/20220324-intel-2021.4.0-omp |
+----------+-------+-----------+------------------------------------+
| 思源一号 | |cpu| | 20210310  | lammps/20210310-intel-2021.4.0-omp |
+----------+-------+-----------+------------------------------------+
| 思源一号 | |gpu| | 20230802  | lammps/20230802-intel-2021.4.0-gpu |
+----------+-------+-----------+------------------------------------+
| pi 2.0   | |cpu| | 20230328  | lammps/20230328-intel-2021.4.0-omp |
+----------+-------+-----------+------------------------------------+
| pi 2.0   | |cpu| | 20220324  | lammps/20220324-oneapi-2021.4.0    |
+----------+-------+-----------+------------------------------------+
| kunpeng  | |arm| | 20190605  | lammps/bisheng-1.3.3-lammps-2019   |
+----------+-------+-----------+------------------------------------+

集群上的 LAMMPS
---------------

- `思源一号 LAMMPS`_

- `π2.0 LAMMPS`_

- `ARM LAMMPS`_


.. _思源一号 LAMMPS:

一. 思源一号 LAMMPS
---------------------

1. 全局部署版本 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本版本支持 intel 加速。对于大部分势函数（eam, lj 等），均推荐使用 intel 加速，计算速度可提升数倍。具体测评和支持范围请见官方文档：`LAMMPS INTEL package <https://docs.lammps.org/Speed_intel.html>`__

使用 intel 加速的 slurm 脚本示例：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=64c512g
   #SBATCH -N 2 
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load lammps/20230328-intel-2021.4.0-omp
   
   mpirun lmp -pk intel 0 omp 2 -sf intel -i in.lj


注意：若体系不支持 intel package，请使用如下 slurm 脚本：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=lmp_test
   #SBATCH --partition=64c512g
   #SBATCH -N 2 
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load lammps/20230328-intel-2021.4.0-omp
   
   mpirun lmp -i in.lj

2. 自行编译 LAMMPS
~~~~~~~~~~~~~~~~~~~~~~~~~~

LAMMPS 自行编译十分容易。下面以在思源一号上为例介绍 LAMMPS 安装

a) 申请计算节点资源用来编译 LAMMPS，并请注意在全部编译结束后退出：

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   
   
b) 从官网获得最新的 LAMMPS，推荐下载最新的版本

.. code:: bash

   wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz

c) 加载 Intel oneapi 模块：

.. code:: bash

   module load oneapi/2021.4.0

d) 编译 (以额外安装 MANYBODY, MEAM, RIGID 和 Intel 加速包为例)

.. code:: bash

   $ tar xvf lammps-stable.tar.gz
   $ cd lammps-XXXXXX
   $ cd src
   $ make                                            #查看编译选项
   $ make package                                    #查看可用的包
   $ make yes-intel yes-manybody yes-meam yes-rigid  #添加所需的包
   $ make ps                                         #查看计划安装的包列表 
   $ make -j 4 oneapi                            #开始编译

e) 环境设置

编译成功后，src 文件夹下将生成可执行文件 lmp_oneapi

为了便于后续调用，一个简单的方法是将该文件移至 ~/bin 文件夹：

.. code:: bash

   $ mkdir ~/bin
   $ cp lmp_oneapi ~/bin

至此安装和设置完成。如下是计算时所需的 slurm 脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lmp
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited
   
   module load oneapi/2021.4.0
   export PATH=~/bin:$PATH

   mpirun lmp_oneapi -pk intel 0 omp 2 -sf intel -i in.lj
   # 若势函数等体系不支持intel加速，则使用下方语句：
   # mpirun lmp_oneapi -i in.lj


.. _π2.0 LAMMPS:

二. π2.0 LAMMPS
----------------

1. Intel编译器部署的版本
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=lammps_pi
   #SBATCH --partition=cpu
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   ulimit -s unlimited
   ulimit -l unlimited
   
   module load lammps/20230328-intel-2021.4.0-omp

   mpirun lmp -pk intel 0 omp 2 -sf intel -i in.lj

2. CPU 版本自行编译
~~~~~~~~~~~~~~~~~~~

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的
Lammps. 下面以在 π 集群为例介绍 lammps 的自行安装

a) 从官网下载 lammps，推荐安装最新的稳定版：

.. code:: bash

   $ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz
   $ or
   $ cp /lustre/share/samples/lammps/lammps-stable.tar.gz ./

b) 由于登录节点禁止运行作业和并行编译，请申请计算节点资源用来编译
   lammps，并在编译结束后退出：

.. code:: bash

   $ srun -p small -n 8 --pty /bin/bash

c) 加载 Intel oneapi 模块：

.. code:: bash

   module load oneapi/2021

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
   $ make -j 8 oneapi    #开始编译

e) 测试脚本

编译成功后，将在 src 文件夹下生成 lmp_oneapi 
后续调用，请给该文件的路径，比如
``~/lammps-3Mar20/src/lmp_oneapi``\ 。脚本名称可设为
slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH -J lammps
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load oneapi/2021

   srun --mpi=pmi2 ~/lammps-3Mar20/src/lmp_oneapi -i in.lj

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
   mpirun -x OMP_NUM_THREADS=1 lmp_aarch64_arm_hypermpi -in in.lj

.. code:: bash

   $ sbatch lammps.slurm

运行结果(单位为：秒，越低越好)
---------------------------------------

思源一号
~~~~~~~~

+------------------------------------------------+
|     lammps/20230328-intel-2021.4.0-omp         |
+=============+==========+===========+===========+
| 核数        | 64       | 128       | 192       |
+-------------+----------+-----------+-----------+
| Wall time   | 0:01:57  | 0:01:01   | 0:00:46   |
+-------------+----------+-----------+-----------+

π2.0
~~~~~

+-----------------------------------------------+
|    lammps/20230328-intel-2021.4.0-omp         |          
+=============+==========+===========+==========+
| 核数        | 40       | 80        | 120      |
+-------------+----------+-----------+----------+
| Wall time   | 0:03:16  | 0:01:35   | 0:01:06  |
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

算例内容如下： `in.lj` 
----------------------------

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



参考资料
--------

-  `LAMMPS 官网 <https://lammps.sandia.gov/>`__
-  `NVIDIA GPU CLOUD <ngc.nvidia.com>`__
