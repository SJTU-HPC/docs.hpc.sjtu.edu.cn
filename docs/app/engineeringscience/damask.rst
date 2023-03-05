.. _damask:

DAMSK
======

简介
----

DAMASK 是一个统一的多物理晶体塑性模拟包。连续体力学边值问题的求解需要连接每个材料点的变形和应力的本构响应,该问题在 DAMASK 中基于晶体可塑性使用各种本构模型和均质化方法能够被有效解决。除此之外，孤立地处理力学已不足以研究新兴的先进高强度材料，在这些材料中，变形的发生与位移相变、显着加热和潜在的损伤演变相关，DAMASK 能够有效处理多物理问题。


可用的版本
-----------

+--------------+-------+----------+---------------------------------------------------------------+
| 版本         | 平台  | 构建方式 | 模块名                                                        |
+==============+=======+==========+===============================================================+
| 3.0.0-alpha5 | |cpu| | spack    | damask/3.0.0-alpha5-gcc-11.2.0-hdf5-openblas-openmpi 思源一号 |
+--------------+-------+----------+---------------------------------------------------------------+
| 3.0.0-alpha6 | |cpu| | spack    | damask/3.0.0-alpha6-gcc-11.2.0-hdf5-openblas-openmpi 思源一号 |
+--------------+-------+----------+---------------------------------------------------------------+
| 3.0.0-alpha7 | |cpu| | spack    | damask/3.0.0-alpha7-gcc-11.2.0-hdf5-openblas-openmpi 思源一号 |
+--------------+-------+----------+---------------------------------------------------------------+
| 3.0.0-alpha5 | |cpu| | spack    | damask/3.0.0-alpha5-gcc-8.3.0-openblas                        |
+--------------+-------+----------+---------------------------------------------------------------+

算例获取方式
-------------

.. code:: bash

   思源：
   mkdir ~/damask && cd ~/damask
   cp -r /dssg/share/sample/damask/* ./
   tar xf grid.tar.xz
   cd grid

   π2.0：
   mkdir ~/damask && cd ~/damask
   cp -r /lustre/share/samples/damask/* ./
   tar xf grid.tar.xz
   cd grid

集群上的DAMASK
--------------------

- `思源一号 DAMASK`_

- `π2.0 DAMASK`_

.. _思源一号 DAMASK:

思源一号上运行DAMASK
-------------------------

3.0.0-alpha7版本的运行脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=32core_damask
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=32
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load damask/3.0.0-alpha7-gcc-11.2.0-hdf5-openblas-openmpi
   mpirun DAMASK_grid --load shearXY.yaml --geom 20grains32x32x32.vti

3.0.0-alpha5版本的运行脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=32core_damask
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=32
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load damask/3.0.0-alpha5-gcc-11.2.0-hdf5-openblas-openmpi
   mpirun DAMASK_grid --load shearXY.yaml --geom 20grains32x32x32.vti

.. _π2.0 DAMASK:

π2.0上运行DAMASK
-------------------------

π2.0上3.0.0-alpha5版本的运行脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=32core_damask
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load damask/3.0.0-alpha5-gcc-8.3.0-openblas
   mpirun -np 32  DAMASK_grid --load shearXY.yaml --geom 20grains32x32x32.vti

运行结果
------------------

单位为秒

思源一号
~~~~~~~~

+-----------------------+
|     3.0.0-alpha6      |
+======+=====+=====+====+
| 核数 | 8   | 16  | 32 |
+------+-----+-----+----+
| 时间 | 210 | 108 | 59 |
+------+-----+-----+----+   

+-----------------------+
|     3.0.0-alpha5      |
+======+=====+=====+====+
| 核数 | 8   | 16  | 32 |
+------+-----+-----+----+
| 时间 | 214 | 109 | 61 |
+------+-----+-----+----+  

π2.0
~~~~~~~~~

+-----------------------+
|     3.0.0-alpha5      |
+======+=====+=====+====+
| 核数 | 8   | 16  | 32 |
+------+-----+-----+----+
| 时间 | 235 | 126 | 78 |
+------+-----+-----+----+

参考链接：https://damask.mpie.de/index.html
