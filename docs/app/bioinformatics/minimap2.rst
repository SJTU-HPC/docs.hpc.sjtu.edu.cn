.. _minimap2:

Minimap2
=============================

简介
----
Minimap2是一个多功能的序列比对程序，
可将DNA或mRNA序列与大型参考数据库进行比对。

可用版本
-------------

+--------+---------+-----------+----------------------------------------------------------+
| 版本   | 平台    | 构建方式  | 模块名                                                   |
+========+=========+===========+==========================================================+
| 2.14   | |cpu|   |spack      | minimap2/2.14-gcc-11.2.0    Pi2.0                        |
+--------+---------+-----------+----------------------------------------------------------+

算例获取方式
--------------

.. code:: bash

   pi2.0：
   mkdir ~/minimap2 && cd ~/minimap2
   cp -r /lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-11.2.0/minimap2-2.14-w7pbon3ooyyiysiroflg3ofh56yhptdj/data/* ./

任务脚本
--------------

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=minimap2
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load minimap2/2.14-gcc-11.2.0
   minimap2 -a MT-human.fa MT-orang.fa > test.sam


参考资料
--------

-  `Minimap2 <https://github.com/lh3/minimap2>`__
