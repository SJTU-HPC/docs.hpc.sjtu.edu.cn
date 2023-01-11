.. _OpenRadioss:

OpenRadioss
====================

简介
----

OpenRadioss is the publicly available open-source code base that a worldwide community of researchers, software developers, and industry leaders are enhancing every day. OpenRadioss is changing the game by empowering users to make rapid contributions that tackle the latest challenges brought on by rapidly evolving technologies like battery development, lightweight materials and composites, human body models and biomaterials, autonomous driving and flight, as well as the desire to give passengers the safest environment possible via virtual testing.




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



参考资料
----------------

-  `OpenRadioss官网 <https://www.openradioss.org/>`__


