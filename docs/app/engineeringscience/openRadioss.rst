.. _OpenRadioss:

OpenRadioss
====================

简介
----

OpenRadioss是一个公开的开源代码库，由全球研究人员、软件开发人员和行业领导者组成的社区每天都在增强它。OpenRadioss正在改变行业规则，让用户能够快速做出贡献，应对快速发展的技术带来的最新挑战，如电池开发、轻质材料和复合材料、人体模型和生物材料、自动驾驶和飞行，以及通过虚拟测试为乘客提供尽可能安全的环境的愿望。有了OpenRadioss，科学家和技术人员可以在专业维护下将研究重点放在稳定的代码库上，该代码库受益于现有有限元功能的大型库以及提供给贡献者的持续集成和持续开发工具。




OpenRadioss安装以及说明
-----------------------------

在思源一号上自行安装OpenRadioss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 首先在思源一号上自己的家目录下新建一个目录作为安装目录,并进入该目录：

.. code::
        
    mkdir OpenRadioss
    cd  OpenRadioss


2. 申请计算资源并加载所需模块：

.. code::
        
    srun -p 64c512g -n 10 --pty /bin/bash
    module load netcdf-fortran/4.5.3-gcc-8.3.1-hdf5-openmpi
    module load openmpi/4.1.1-gcc-9.3.0
    module load gcc/9.3.0


3.  从  `OpenRadioss官网 <https://www.openradioss.org/>`__  下载Source code(tar.gz)到自己本地计算机并上传到刚才创建的目录下(可借助filezilla或者超算可视化平台传输文件)：


4. 解压上传的压缩文件并进入该目录：

.. code::

  tar -xzvf OpenRadioss-latest-20221205.tar.gz
  cd OpenRadioss-latest-20221205


5. 进入starter目录并开始第一步编译：

.. code::

  cd starter
  ./build_script.sh -arch=linux64_gf


6. 退出starter目录，进入engine目录，开始第二步编译：

.. code::

  cd ../engine
  ./build_script.sh -arch=linux64_gf -mpi=ompi  -mpi-root=/dssg/opt/icelake/linux-centos8-icelake/gcc-9.3.0/openmpi-4.1.1-usre7vgur4rv6jllqd4yuf5gg57kothm -mpi-include=/dssg/opt/icelake/linux-centos8-icelake/gcc-9.3.0/openmpi-4.1.1-usre7vgur4rv6jllqd4yuf5gg57kothm/include -mpi-libdir=/dssg/opt/icelake/linux-centos8-icelake/gcc-9.3.0/openmpi-4.1.1-usre7vgur4rv6jllqd4yuf5gg57kothm/lib








在pi2.0上自行安装OpenRadioss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；
2. 申请计算资源并加载所需模块：

.. code::
        
  srun -p cpu -N  1 --ntasks-per-node 40  --exclusive  --pty /bin/bash
  module load netcdf-fortran/4.5.2-gcc-8.3.0-openmpi
  module load openmpi/4.0.5-gcc-9.2.0
  module load gcc/9.2.0


3. 此步骤和上文完全相同；
4. 此步骤和上文完全相同；
5. 此步骤和上文完全相同；

6. 退出starter目录，进入engine目录，开始第二步编译：

.. code::

  cd ../engine
  ./build_script.sh -arch=linux64_gf -mpi=ompi  -mpi-root=/lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/openmpi-4.0.5-vpswzpisyoc6gl6e5isbal66yykxdc6k -mpi-include=/lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/openmpi-4.0.5-vpswzpisyoc6gl6e5isbal66yykxdc6k/include -mpi-libdir=/lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/openmpi-4.0.5-vpswzpisyoc6gl6e5isbal66yykxdc6k/lib

在pi2.0 kos系统上自行安装OpenRadioss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 下载并解压源代码文件，申请计算节点

.. code::
        
  wget https://github.com/OpenRadioss/OpenRadioss/archive/refs/tags/latest-20231220.tar.gz
  tar zxvf latest-20231220.tar.gz
  cd OpenRadioss-latest-20231220/
  srun -p cpu -n 8 --pty /bin/bash #申请计算节点

2. 加载编译所需模块：

.. code::
        
  module load openmpi/4.1.5-gcc-8.5.0
  module load netcdf-fortran/4.5.2-gcc-8.5.0-openmpi
  module load cmake/3.26.3-gcc-8.5.0
  module load python/3.10.10-gcc-8.5.0

3. 第一步编译
  
.. code::

  cd starter
  ./build_script.sh -arch=linux64_gf

4. 第二步编译

.. code::

  cd ../engine
  ./build_script.sh -arch=linux64_gf -mpi=ompi  -mpi-root=/lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/openmpi-4.1.5-sjnibarr4bwsceb3ncopdeyfigesuzfk/ -mpi-include=/lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/openmpi-4.1.5-sjnibarr4bwsceb3ncopdeyfigesuzfk/include -mpi-libdir=/lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/openmpi-4.1.5-sjnibarr4bwsceb3ncopdeyfigesuzfk/lib


参考资料
----------------

-  `OpenRadioss官网 <https://www.openradioss.org/>`__


