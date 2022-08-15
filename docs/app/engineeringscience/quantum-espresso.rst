.. _quantum-espresso:

Quantum ESPRESSO
================

简介
----

Quantum ESPRESSO基于密度泛函理论、平面波和赝势（范数守恒和超软）开发，是用于纳米级电子结构计算和材料建模的开源软件包。

根据GNU通用公共许可证的条款，全世界的研究人员均可免费使用。

可用的版本
----------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 7.1    | |cpu|   | spack    | quantum-espresso/7.1-intel-2021.4.0 思源一号              |
+--------+---------+----------+-----------------------------------------------------------+
| 7.0    | |cpu|   | spack    | quantum-espresso/7.0-intel-2021.4.0 思源一号              |
+--------+---------+----------+-----------------------------------------------------------+
| 6.7    | |cpu|   | spack    | quantum-espresso/6.7-intel-2021.4.0 思源一号              |
+--------+---------+----------+-----------------------------------------------------------+
| 6.7    | |cpu|   | spack    | quantum-espresso/6.7-gcc-11.2.0-openblas-openmpi 思源一号 |
+--------+---------+----------+-----------------------------------------------------------+
| 7.1    | |cpu|   | spack    | quantum-espresso/7.1-intel-2021.4.0                       |
+--------+---------+----------+-----------------------------------------------------------+
| 7.0    | |cpu|   | spack    | quantum-espresso/7.0-intel-2021.4.0                       |
+--------+---------+----------+-----------------------------------------------------------+
| 6.6    | |cpu|   | 容器     | quantum-espresso/6.6                                      |
+--------+---------+----------+-----------------------------------------------------------+
| 6.7    | |cpu|   | 源码编译 | quantum-espresso/6.7-intel-21.4.0-impi                    |
+--------+---------+----------+-----------------------------------------------------------+

Spack安装参考
--------------

.. code:: bash

   spack install quantum-espresso@7.1%intel@2021.4.0 +libxc ^intel-oneapi-mpi

算例下载
---------

.. code:: bash

   wget https://repository.prace-ri.eu/git/UEABS/ueabs/-/raw/master/quantum_espresso/test_cases/small/ausurf.in
   wget https://repository.prace-ri.eu/git/UEABS/ueabs/-/raw/master/quantum_espresso/test_cases/small/Au.pbe-nd-van.UPF

集群上的Quantum ESPRESSO
------------------------

- `思源一号`_
 
- `π2.0集群`_

.. _思源一号:

一. 思源一号上的Quantum ESPRESSO
--------------------------------

基于intel编译器编译的版本
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=1node_qe
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi
   module load quantum-espresso/6.7-intel-2021.4.0
   
   export OMP_NUM_THREADS=1
   ulimit -s unlimited
   ulimit -l unlimited
   
   mpirun pw.x -i ausurf.in

使用如下脚本提交作业

.. code:: bash

   sbatch qe_intel.slurm

运行结果如下所示

.. code:: bash

   PWSCF        :   3m50.28s CPU   3m53.80s WALL

   tree out
   out/
   ├── ausurf.save
   │   ├── Au.pbe-nd-van.UPF
   │   ├── charge-density.dat
   │   ├── data-file-schema.xml
   │   ├── wfc1.dat
   │   └── wfc2.dat
   └── ausurf.xml

基于GCC编译器编译的版本
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=1node_qe_gcc
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load openmpi/4.1.1-gcc-11.2.0
   module load quantum-espresso/6.7-gcc-11.2.0-openblas-openmpi
   
   export OMP_NUM_THREADS=1
   ulimit -s unlimited
   ulimit -l unlimited
   
   mpirun pw.x -i ausurf.in

使用如下命令提交作业

.. code:: bash

   sbatch qe_gcc.slurm

运行结果如下所示：

.. code:: bash

   PWSCF        :   5m18.95s CPU   5m26.66s WALL

   tree out
   out/
   ├── ausurf.save
   │   ├── Au.pbe-nd-van.UPF
   │   ├── charge-density.dat
   │   ├── data-file-schema.xml
   │   ├── wfc1.dat
   │   └── wfc2.dat
   └── ausurf.xml
   
   1 directory, 6 files

.. _π2.0集群:

二. π2.0集群上的Quantum ESPRESSO
--------------------------------

基于intel2021.4.0编译器编译的6.7版本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH -J 80cores
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   
   ulimit -s unlimited
   ulimit -l unlimited
   module load quantum-espresso/6.7-intel-21.4.0-impi
   
   mpirun pw.x -i ausurf.in

使用如下命令提交作业

.. code:: bash

   sbatch qe_intel.slurm

运行结果如下所示：

.. code:: bash

   PWSCF        :   6m42.48s CPU   6m53.24s WALL

使用容器部署的版本
~~~~~~~~~~~~~~~~~~

在 cpu 队列上，总共使用 80 核 (n = 80) cpu 队列每个节点配有 40
核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash

   #SBATCH -J QE_test
   #SBATCH -p cpu
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load quantum-espresso

   srun --mpi=pmi2 pw.x -i ausurf.in

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

运行结果如下所示：

.. code:: bash

   PWSCF        :  17m37.92s CPU  17m51.67s WALL

   tree out
   out/
       ├── ausurf.save
       │   ├── Au.pbe-nd-van.UPF
       │   ├── charge-density.dat
       │   ├── data-file-schema.xml
       │   ├── wfc1.dat
       │   └── wfc2.dat
       └── ausurf.xml

运行结果
--------

思源一号
~~~~~~~~

+--------------------------------------------+
|    quantum-espresso/6.7-intel-2021.4.0     |
+===========+==========+==========+==========+
| 核数      | 64       | 128      | 192      |
+-----------+----------+----------+----------+
| CPU time  | 5m32.13s | 3m49.22s | 3m41.00s |
+-----------+----------+----------+----------+

+--------------------------------------------------+
| quantum-espresso/6.7-gcc-11.2.0-openblas-openmpi |
+===========+============+============+============+
| 核数      | 64         | 128        | 192        |
+-----------+------------+------------+------------+
| CPU time  | 6m44.78s   | 5m18.95s   | 5m31.64s   |
+-----------+------------+------------+------------+

π2.0
~~~~~

+-----------------------------------------------+
|    quantum-espresso/6.7-intel-21.4.0-impi     |
+===========+===========+===========+===========+
| 核数      | 40        | 80        | 120       |
+-----------+-----------+-----------+-----------+
| CPU time  | 9m21.27s  | 6m42.48s  | 5m 1.21s  |
+-----------+-----------+-----------+-----------+

+-----------------------------------------------+
|             quantum-espresso/6.6              |
+===========+===========+===========+===========+
| 核数      | 40        | 80        | 120       |
+-----------+-----------+-----------+-----------+
| CPU time  | 19m27.24s | 17m39.15s | 15m25.99s |
+-----------+-----------+-----------+-----------+

建议
----

通过分析上述运行结果，我们推荐您使用以下两个版本运行QE作业

.. code:: bash

   module load quantum-espresso/6.7-intel-2021.4.0       思源一号
   module load quantum-espresso/6.7-intel-21.4.0-impi    π2.0

参考资料
--------

-  `Quantum ESPRESSO 官网 <https://www.quantum-espresso.org/>`__
