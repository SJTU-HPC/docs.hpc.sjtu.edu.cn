.. _lobster:

lobster
============

软件简介
------------
Lobster 是一种用于计算局域轨道基态轨道结合能的电子结构分析软件工具，主要用于分析固体材料中的化学键和电子结构。它的名称“Lobster”是“Local Orbital Basis Suite Towards Electronic-Structure Reconstruction”的缩写。

可用的版本
-----------

+--------+---------+----------+----------+-------------------------------------------------+
| 版本   | 平台    | 构建方式 |集群      | 模块名                                          |
+========+=========+==========+==========+=================================================+
| 5.1.0  | |cpu|   | 源码     | Pi2.0    |lobster/5.1-gcc-8.5.0                            |
+--------+---------+----------+----------+-------------------------------------------------+
| 5.1.0  | |cpu|   | 源码     | 思源一号 |lobster/5.1-gcc-8.5.0                            |
+--------+---------+----------+----------+-------------------------------------------------+


作业脚本示例
------------
思源一号
~~~~~~~~
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
    module load vasp/5.4.4-intel-2021.4.0
    module load lobster/5.1-gcc-8.5.0

    ulimit -s unlimited
    mpirun vasp_std

    export OMP_NUM_THREADS=64
    lobster-5.1.0

Pi 2.0
~~~~~~~
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
    module load lobster/5.1-gcc-8.5.0

    ulimit -s unlimited
    mpirun vasp_std

    export OMP_NUM_THREADS=40
    lobster-5.1.0

软件简单使用流程
----------------


获取算例
~~~~~~~~~~~
.. code:: bash
    
    module load lobster/5.1.0
    cp -rfv $lobster_vasp_example .
    cd fullerene

算例介绍
~~~~~~~~~~~~~~~~~~~
以官方提供的VASP算例的富勒烯（fullerene）为例，官方算例已提供VASP算例所需的计算输入文件，需要先使用VASP进行DFT计算，然后再进行LOBSTER的计算，下面示例为在思源一号使用1节点64核心进行计算，由于LOBSTER不支持跨节点并行，需要设置OMP线程参数。

准备作业脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    module load vasp/5.4.4-intel-2021.4.0
    module load lobster/5.1-gcc-8.5.0

    ulimit -s unlimited
    mpirun vasp_std

    export OMP_NUM_THREADS=64
    lobster-5.1.0


提交作业
~~~~~~~~~~~
.. code:: console

    sbatch run.slurm

查看结果
~~~~~~~~~~~
.. code:: bash

    writing SitePotentials.lobster and MadelungEnergies.lobster...
    finished in 0 h 10 min 14 s 484 ms of wall time
                10 h  9 min 37 s 340 ms of user time
                0 h 10 min 33 s 560 ms of sys  time


参考资料
--------

-  `LOBSTER <http://www.cohp.de/>`__
