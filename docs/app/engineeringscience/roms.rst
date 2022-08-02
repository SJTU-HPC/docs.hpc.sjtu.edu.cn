.. _roms:

ROMS
=====

简介
------

ROMS全称Regional Ocean Modeling System，是一个三维区域海洋模型，由罗格斯大学海洋与海岸科学研究所和加利福尼亚大学洛杉矶分校共同研究开发，被广泛应用于海洋及河口地区的水动力及水环境模拟。

ROMS3.6依赖的软件及其版本
-------------------------

思源一号和π2.0上安装的ROMS、依赖软件及其版本保持一致

.. code:: bash

   zlib           : 1.2.11
   szip           : 2.1.1
   hdf5           : 1.12.0
   netcdf-c       : 4.8.0
   netcdf-fortran : 4.5.3
   pnetcdf        : 1.12.0
   parallelIO     : 2.5.6

使用module调用ROMS依赖环境
----------------------------

.. code:: bash

   思源一号 : module load roms/3.6-intel-2021.4.0
   π2.0    : module load roms/3.6-intel-2021.4.0

依赖软件具体的安装路径

.. code:: bash

   思源一号
   zlib          : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/netcdf
   szip          : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/netcdf
   hdf5          : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/netcdf
   netcdf-c      : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/netcdf
   netcdf-fortran: /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/netcdf
   parallelNetcdf: /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/pnetcdf
   parallelIO    : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMSRely/pio
   
   π2.0
   zlib          : /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/netcdf
   szip          : /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/netcdf
   hdf5          : /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/netcdf
   netcdf-c      : /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/netcdf
   netcdf-fortran: /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/netcdf
   parallelNetcdf: /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/pnetcdf
   parallelIO    : /lustre/opt/contribute/cascadelake/ROMS/ROMSRely/pio


集群上如何使用ROMS
--------------------

- `思源一号 自定义编译ROMS`_

- `π2.0 自定义编译ROMS`_

- `思源一号 预编译ROMS`_

- `π2.0 预编译ROMS`_

.. _思源一号 自定义编译ROMS:

思源一号 自定义编译ROMS
--------------------------

为方便广大师生编译ROMS，我们已将 ```build_roms.sh``` 和 ```Linux-ifort.mk``` 两个文件中依赖软件的参数提前修改
上述两个文件的位置为

.. code:: bash
    
   /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMS_3.6/ROMS/Bin/build_roms.sh
   /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMS_3.6/Compilers/Linux-ifort.mk

.. code:: bash

   srun -p 64c512g -N 1 --exclusive --pty /bin/bash
   mkdir -p ~/ROMS/ROMSProjects/upwelling
   export PROJECT_DIR=~/ROMS/ROMSProjects/upwelling
   cd $PROJECT_DIR
   cp /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMS_3.6/ROMS/Bin/build_roms.sh ./
   cp /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMS_3.6/ROMS/Include/upwelling.h ./
   cp /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMS_3.6/ROMS/External/roms_upwelling.in ./ 
   ./build_roms.sh

.. _π2.0 自定义编译ROMS:

π2.0 自定义编译ROMS
----------------------

为方便广大师生编译ROMS，我们已将 ```build_roms.sh``` 和 ```Linux-ifort.mk``` 两个文件中依赖软件的参数提前修改
上述两个文件的位置为

.. code:: bash
    
   /lustre/opt/contribute/cascadelake/ROMS/ROMS_3.6/ROMS/Bin/build_roms.sh
   /lustre/opt/contribute/cascadelake/ROMS/ROMS_3.6/Compilers/Linux-ifort.mk

.. code:: bash

   srun -p cpu -N 1 --exclusive --pty /bin/bash
   mkdir -p ~/ROMS1/ROMSProjects/upwelling
   export PROJECT_DIR=~/ROMS1/ROMSProjects/upwelling
   cd $PROJECT_DIR
   cp /lustre/opt/contribute/cascadelake/ROMS/ROMS_3.6/ROMS/Bin/build_roms.sh ./
   cp /lustre/opt/contribute/cascadelake/ROMS/ROMS_3.6/ROMS/Include/upwelling.h ./
   cp /lustre/opt/contribute/cascadelake/ROMS/ROMS_3.6/ROMS/External/roms_upwelling.in ./ 
   ./build_roms.sh

.. _思源一号 预编译ROMS:

思源一号 预编译ROMS
----------------------

更改 ```roms_upwelling.in``` 文件的参数，如下所示

.. code:: bash

   VARNAME = /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/ROMS/ROMS_3.6/ROMS/External/varinfo.yaml
   NtileI == 2                               ! I-direction partition
   NtileJ == 2                               ! J-direction partition

```NtileI``` 和 ```NtileJ``` 的乘积需等于总核数

提交如下脚本运行作业

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=ROMS
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load roms/3.6-intel-2021.4.0
   mpirun -np 4 romsM roms_upwelling.in

   
.. _π2.0 预编译ROMS:

π2.0 预编译ROMS
----------------------

更改 ```roms_upwelling.in``` 文件的参数，如下所示

.. code:: bash

   VARNAME = /lustre/opt/contribute/cascadelake/ROMS/ROMS_3.6/ROMS/External/varinfo.yaml
   NtileI == 2                               ! I-direction partition
   NtileJ == 2                               ! J-direction partition

```NtileI``` 和 ```NtileJ``` 的乘积需等于总核数

提交如下脚本运行作业

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=ROMS
   #SBATCH --partition=small
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load roms/3.6-intel-2021.4.0
   mpirun -np 4 romsM roms_upwelling.in


   
运行结果
------------

思源一号上的ROMS
~~~~~~~~~~~~~~~~~

+------+-----+----+----+----+
| 核数 | 1   | 2  | 4  | 8  |
+======+=====+====+====+====+
| 时间 | 107 | 56 | 36 | 23 |
+------+-----+----+----+----+

π2.0上的ROMS
~~~~~~~~~~~~~~~~~
  
+------+-----+----+----+----+
| 核数 | 1   | 2  | 4  | 8  |
+======+=====+====+====+====+
| 时间 | 134 | 70 | 41 | 29 |
+------+-----+----+----+----+

参考资料
--------

-  `ROMS官方网站 <https://www.myroms.org/>`__
