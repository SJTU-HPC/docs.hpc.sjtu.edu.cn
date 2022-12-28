Osiris
======

简介
----

Osiris是一个在完全相对论性下的，大规模并行计算胞内粒子（PIC）的程序。

软件安装
----------

此软件是开源项目，但相关git仓库并不支持公开访问，因此需要申请权限。软件仓库地址：https://github.com/GoLP-IST/osiris

1.将需要的软件分支版本克隆至超算集群中
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/GoLP-IST/osiris.git

2.修改配置
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
  
   vim osiris/config/osiris_sys.linux.gnu
   
检查OPENMPI_ROOT,SZIP_ROOT,H5_ROOT这三个路径是否与.bashrc文件中的一致，如果不一致则改为一致

3.分维度进行编译
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
  
   cd osiris
   chmod 777 configure
   ./configure -t production -s linux.gnu -d 1/2/3 #（其中，1/2/3表示维度，三个维度分别编译）
   make

4.确认编译结果
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
  
   ls bin
   #可以见到类似osiris-r-2D.e的可执行程序
   chmod 755 osiris-r-2D.e

