.. _namd:

NAMD  
============

简介
-------

NAMD (Nanoscale Molecular Dynamics) 是一个高性能并行分子动力学模拟软件，广泛用于模拟生物大分子的运动，如蛋白质、核酸和膜。它的设计目标是能够处理非常大的系统，并且支持多种计算平台，从桌面工作站到超级计算机。NAMD 通常与 VMD (Visual Molecular Dynamics) 配合使用，用于前处理和可视化模拟结果。

可用的版本
-----------

+--------+---------+----------+----------+-----------------------------------------------------+
| 版本   | 平台    | 构建方式 | 集群     | 模块名                                              |
+========+=========+==========+==========+=====================================================+
| 3.0b6  | |cpu|   | 源码     | Pi 2.0   |namd/3.0b6-gcc-11.2.0                                |
+--------+---------+----------+----------+-----------------------------------------------------+
| 3.0b6  | |cpu|   | 源码     | 思源一号 |namd/3.0b6-gcc-11.2.0                                |
+--------+---------+----------+----------+-----------------------------------------------------+
| 3.0b6  | |arm|   | 源码     | ARM      |namd/3.0b6-gcc-10.3.1-openmpi                        |
+--------+---------+----------+----------+-----------------------------------------------------+

作业脚本示例
------------
思源一号
~~~~~~~~
.. code:: bash
    
    #!/bin/bash

    #SBATCH --job-name=namd
    #SBATCH --partition=64c512g
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=64
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load namd/3.0b6-gcc-11.2.0

    mpirun -np $SLURM_NPROCS namd3 apoa1.namd

Pi 2.0
~~~~~~~
.. code:: bash

    #!/bin/bash
  
    #SBATCH --job-name=namd
    #SBATCH --partition=cpu
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=40
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load namd/3.0b6-gcc-11.2.0

    mpirun -np $SLURM_NPROCS namd3 apoa1.namd

ARM
~~~~~~
.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=namd
    #SBATCH --partition=arm128c256g
    #SBATCH -N 2
    #SBATCH --ntasks-per-node=128
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load namd/3.0b6-gcc-10.3.1-openmpi 

    mpirun -np $SLURM_NPROCS namd3 apoa1.namd

软件使用流程
----------------

获取算例
~~~~~~~~~~~
.. code:: bash
    
    wget https://www.ks.uiuc.edu/Research/namd/utilities/apoa1.tar.gz
    tar -xvf apoa1.tar.gz
    cd apoa1

算例介绍
~~~~~~~~~~~~~~~~~~~~
ApoA1 benchmark 是 NAMD 中一个常用的基准测试算例，常用于评估高性能计算平台上 NAMD 的计算性能。这个基准测试模拟了载脂蛋白 A-I (Apolipoprotein A-I, ApoA1) 的分子动力学，ApoA1 是高密度脂蛋白 (HDL) 的主要蛋白质成分，与胆固醇代谢和心血管健康密切相关。

系统组成: ApoA1 benchmark 系统包含 92,224 个原子，模型中包括一个ApoA1蛋白质环绕着 3000 个水分子，以及 120 个脂肪酸分子。这些分子构成了一个水溶液环境中的 HDL 磷脂双分子层。


准备作业脚本，以思源一号为例：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash
    
    #!/bin/bash

    #SBATCH --job-name=namd
    #SBATCH --partition=64c512g
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=64
    #SBATCH --exclusive
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load namd/3.0b6-gcc-11.2.0

    mpirun -np $SLURM_NPROCS namd3 apoa1.namd


提交作业
~~~~~~~~~~~
.. code:: console

    sbatch run.slurm


查看结果
~~~~~~~~~~~
.. code:: bash
    
    WRITING EXTENDED SYSTEM TO OUTPUT FILE AT STEP 500
    WRITING COORDINATES TO OUTPUT FILE AT STEP 500
    The last position output (seq=-2) takes 0.170 seconds, 59.409 MB of memory in use
    WRITING VELOCITIES TO OUTPUT FILE AT STEP 500
    The last velocity output (seq=-2) takes 0.051 seconds, 59.409 MB of memory in use
    ====================================================

    WallClock: 30.385077  CPUTime: 30.385077  Memory: 59.408632 MB
    [Partition 0][Node 0] End of program

参考资料
--------

-  `NAMD 官网 <https://www.ks.uiuc.edu/Research/namd/>`__
