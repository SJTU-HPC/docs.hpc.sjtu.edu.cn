.. _namd:

NAMD  
============

简介
-------
NAMD is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems. Based on Charm++ parallel objects, NAMD scales to hundreds of cores for typical simulations and beyond 500,000 cores for the largest simulations. NAMD uses the popular molecular graphics program VMD for simulation setup and trajectory analysis, but is also file-compatible with AMBER, CHARMM, and X-PLOR. NAMD is distributed free of charge with source code.

可用的版本
-----------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 3.0b3  | |cpu|   | spack    | namd/3.0b3-gcc-8.5.0 Pi2.0-KOS                            |
+--------+---------+----------+-----------------------------------------------------------+
| 3.0b3  | |cpu|   | spack    | namd/3.0b3-intel-2021.4.0  思源一号                       |
+--------+---------+----------+-----------------------------------------------------------+


使用spack在集群上安装NAMD
--------------------------------------

以在Pi2.0集群安装为例，需要先在NAMD官网注册后下载NAMD对应版本的安装包，上传到家目录

.. code:: console
    
    $ srun -p cpu -n 8 --pty /bin/bash
    $ spack spec namd@3.0b3 %gcc@8.5.0
    $ spack install namd@3.0b3 %gcc@8.5.0


参考资料
--------

-  `NAMD <https://www.ks.uiuc.edu/Research/namd/>`__
