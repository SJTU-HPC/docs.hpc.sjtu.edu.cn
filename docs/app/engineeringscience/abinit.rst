ABINIT
======

简介
----

ABINIT是一个软件套件，可用于计算材料的光学、机械、振动和其他可观察特性的模拟。从密度泛函理论的量子方程开始，可以使用基于DFT的微扰理论和多体格林函数（GW 和 DMFT）建立高级应用程序。
ABINIT可以计算具有任何化学成分的分子、纳米结构和固体，并附带几个完整且强大的原子势表。

可用版本
--------

+-------+-------+----------+-------------------------------------------------------+
| 版本  | 平台  | 构建方式 | 模块名                                                |
+=======+=======+==========+=======================================================+
| 9.4.2 | |cpu| | spack    | abinit/9.4.2-gcc-8.3.1-hdf5-openblas-openmpi 思源一号 |
+-------+-------+----------+-------------------------------------------------------+

算例位置
----------

.. code:: bash

   /dssg/share/sample/abinit/abinit-9.6.2.tar.gz

集群上的ABINIT
-------------------

- `思源一号上运行ABINIT`_

.. _思源一号上运行ABINIT:

思源一号上运行ABINIT
---------------------

拷贝算例到本地
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   mkdir ~/abinit && cd ~/abinit
   cp -r /dssg/share/sample/abinit/abinit-9.6.2.tar.gz ./
   tar xf abinit-9.6.2.tar.gz

运行脚本
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=abinit
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   export ABI_HOME=~/abinit/abinit-9.6.2
   export ABI_TESTS=$ABI_HOME/tests/
   export ABI_PSPDIR=$ABI_TESTS/Psps_for_tests/
   module load abinit
   module load openmpi/4.1.1-gcc-8.3.1 
   mpirun -np 8 abinit tbs_1.abi


运行结果如下所示：
-------------------

思源一号上ABINIT的运行时间
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------------------------------------+
|      abinit/9.4.2-gcc-8.3.1-hdf5-openblas-openmpi        | 
+=============+==========+===========+===========+=========+
| 核数        | 1        | 2         | 4         | 8       |
+-------------+----------+-----------+-----------+---------+
| Exec time   | 0:00:29  | 0:00:16   | 0:00:11   | 0:00:07 |  
+-------------+----------+-----------+-----------+---------+

参考资料
--------

-  ABINIT http://www.abinit.org
