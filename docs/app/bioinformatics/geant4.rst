.. _Geant4:

GEANT4
======

简介
----
Geant4(GEometry ANd Tracking，几何和跟踪)是由CERN(欧洲核子研究组织)基于C++面向对象技术开发的蒙特卡罗应用软件包，
用于模拟粒子在物质中输运的物理过程。相对于MCNP、EGS等商业软件来说,它的主要优点是源代码完全开放,用户可以根据实际需要更改、
扩充Geant4程序。详情请查阅 `Geant4官网 <https://cern.ch/geant4>`_

思源一号自定义编译Geant4
--------------------------

- 申请计算节点并加载模块

.. code:: bash

    srun -p 64c512g -n 4 --pty /bin/bash
    module load cmake/3.21.4-gcc-11.2.0

- 下载源码

.. code:: bash

    git clone https://github.com/Geant4/geant4.git
    cd geant4

- 编译。假定解压后的源文件所在路径为(path_to_source_code)，软件需要安装到路径(path_to_your_installation)

.. code:: bash

    cmake -S ./ -B /(path_to_your_installation)
    make && make install

- 激活Geant4数据集

.. code:: bash

    cd (path_to_your_installation)
    cmake -DGEANT4_INSTALL_DATA=ON .
    make && make install

- 激活环境变量

.. code:: bash

    source /(path_to_your_installation)/bin/geant4.sh

制作Geant4可执行程序
--------------------------

本文档使用cmake构建示例Geant4应用程序，其中源文件与脚本位于(path_to_your_installation)/share/Geant4-11.0.3/examples/basic/B1路径下

- 建立用于编译的目录B1_example_build并编译示例

.. code:: bash

    mkdir B1_example_build
    cd B1_example_build
    cmake -DGeant4_DIR=/(path_to_your_installation)/lib64/Geant4-11.0.3 /(path_to_your_installation)/share/Geant4-11.0.3/examples/basic/B1
    make -j

- 执行应用程序exampleB1。经过上一步编译后将得到名称为exampleB1的可执行文件，然后执行该文件

.. code:: bash

    ./exampleB1