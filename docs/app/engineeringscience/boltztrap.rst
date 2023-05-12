.. _boltztrap:

BoltzTraP
===========

简介
----

BoltzTraP（Boltzmann Transport Properties）是一个计算半经典输运系数的程序。

软件下载链接
--------------

.. code:: bash
   
   #包含算例
   https://owncloud.tuwien.ac.at/index.php/s/s2d55LYlZnioa3s/download 

算例所在目录

.. code:: bash

   /path/download/boltztrap-1.2.5/tests

集群上的BoltzTraP
--------------------

- `思源一号 BoltzTraP`_

- `π2.0 BoltzTraP`_

.. _思源一号 BoltzTraP:

思源一号上的BoltzTraP
-------------------------------------

运行脚本如下所示：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=boltztrap
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load BoltzTraP 
   #使用tests目录下的Al算例
   BoltzTraP BoltzTraP.def 


.. _π2.0 BoltzTraP:

π2.0上的BoltzTraP
-------------------------------------

运行脚本如下所示：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=BoltzTraP
   #SBATCH --partition=small
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load BoltzTraP 
   #使用tests目录下的Al算例
   BoltzTraP BoltzTraP.def

运行结果(单位为：s)
---------------------

.. code:: bash

    ================ BoltzTraP vs 1.2.5 =============
    Al                                                                             
     XXXXXXXX
      1.68966258649651      -0.503059854949352       5.000000000000000E-004 npoints
           4386

参考资料
--------

-  `BoltzTraP 官网 <http://www.icams.de/content/departments/cmat/boltztrap/>`__
