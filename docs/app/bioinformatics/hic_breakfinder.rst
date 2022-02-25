.. _hic_breakfinder:

Hic_breakfinder
================

简介
----

a framework that integrates optical mapping, high-throughput chromosome conformation capture (Hi-C), and whole genome sequencing to systematically detect SVs in a variety of normal or cancer samples and cell lines.

可用的版本
----------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 1.0    | |cpu|   | 源码编译 | hic_breakfinder/1.0-gcc-9.2.0                             |
+--------+---------+----------+-----------------------------------------------------------+

数据下载
--------

.. code:: bash

   https://salkinstitute.box.com/s/m8oyv2ypf8o3kcdsybzcmrpg032xnrgx


运行示例
--------

官方提供的数据下载比较慢，作业脚本中 ``${Example}`` 存放了示例数据。

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=hic_breakfinder
   #SBATCH --partition=small
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=5

   module load hic_breakfinder/1.0-gcc-9.2.0
   hic_breakfinder --bam-file ${Example}/K562_in_house_b38d5.nodup.bam --exp-file-inter ${Example}/inter_expect_1Mb.hg38.txt --exp-file-intra ${Example}/intra_expect_100kb.hg38.txt --min-1kb --name Out

使用如下脚本提交作业

.. code:: bash

   sbatch test.slurm

运行结果见 ``Out.breaks.txt``

运行资源情况
------------

.. code:: bash

   Cluster: sjtupi
   State: COMPLETED (exit code 0)
   Nodes: 1
   Cores per node: 5
   CPU Utilized: 07:57:02
   CPU Efficiency: 19.97% of 1-15:48:35 core-walltime
   Job Wall-clock time: 07:57:43
   Memory Utilized: 4.68 GB
   Memory Efficiency: 23.38% of 20.00 GB

.. tip:: Hic_breakfinder是单线程的程序，不能有效利用多核资源，建议大家根据内存占用情况来申请资源。

参考资料
--------

-  `hic_breakfinder <https://github.com/dixonlab/hic_breakfinder>`__
