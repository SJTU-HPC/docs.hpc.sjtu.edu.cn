.. _cesm:

CESM
=====

简介
------

CESM（Community Earth System Model）是一个完全耦合的全球气候模型，可用于地球过去、现在和未来气候状态的模拟。

CESM源码包的位置
-------------------

.. code:: bash

   思源一号 ：/dssg/share/data/cesm/my_cesm_sandbox.tar.gz

测试数据的位置
-----------------

.. code:: bash

   思源一号 ：/dssg/share/data/cesm/inputdata/ 


思源一号上CESM的使用
----------------------

导入CESM依赖环境
~~~~~~~~~~~~~~~~~~~~~~

用户需要在自己的目录下编辑CESM才能正常使用，我们已在思源上部署了CESM的依赖软件：

+------+--------------------------------+
| 版本 | 调用方式                       |
+======+================================+
| 2.2  | module load cesm/2.2-gcc-8.3.1 |
+------+--------------------------------+

上述模块中包含的依赖软件为：

.. code:: bash

   HDF5 : 1.10.1
   OpenBLAS: 0.3.20
   netcdf-c : 4.4.1.1
   netcdf-fortran : 4.4.1
   parallel-netcdf : 1.9.0
   zlib : 1.2.10

个人用户目录下构建CESM执行环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

申请计算节点与导入执行环境

.. code:: bash

   srun -p 64c512g --exclusive --pty /bin/bash
   module load cesm

.. code:: bash

   mkdir ~/CESM2.2 && cd ~/CESM2.2
   cp /dssg/share/data/cesm/my_cesm_sandbox.tar.gz ./
   tar xf my_cesm_sandbox.tar.gz
   cd my_cesm_sandbox/cime/scripts/
   ./create_newcase --case test_12 --compset X --res f19_g16 --mach=centos7-linux
   cd test_12/
   ./case.setup
   cp -r /dssg/share/data/cesm/Macros.make ./
   ./case.build --skip-provenance-check
   
执行作业
~~~~~~~~~~~

作业脚本

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=cesm
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load cesm
   ulimit -s unlimited
   ./case.submit
   
运行结果
------------

思源一号上的CESM
~~~~~~~~~~~~~~~~~

.. code:: bash

   copying file /dssg/home/acct-hpc/hpc/cesm/scratch/mycase_single/run/mycase_single.cpl.r.0001-01-06-00000.nc to /dssg/home/acct-hpc/hpc/cesm/archive/mycase_single/rest/0001-01-06-00000/mycase_single.cpl.r.0001-01-06-00000.nc
   Archiving restarts for dart (esp)
   Archiving history files for drv (cpl)
   Archiving history files for dart (esp)
   st_archive completed
   Submitted job case.run with id None
   Submitted job case.st_archive with id None
  
参考资料
--------

-  `CESM官方网站 <https://http://www.cesm.ucar.edu/>`__
-  `CESM User
   Guide <http://www.cesm.ucar.edu/models/cesm1.2/cesm/doc/usersguide/book1.html>`__
