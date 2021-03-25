#

.. raw:: html

   <center>

CHARMM

.. raw:: html

   </center>

--------------

CHARMM
======

本文档向您展示如何通过源码在家目录中安装CHARMM软件

简介
----
CHARMM于70年代末诞生于 Martin Karplus 小组，其前身正是历史上首次尝试基于蛋白质结构计算能量所使用的程序。该程序最初基于 Lifson 的 consistent force field (CCF)，其后由 Bruce Gelin 和 Andy MacCammon 等发展，成为从结构到相互作用再到动力学模拟的一套方法。

该软件集成了生物大分子动力学模拟领域的各种前沿算法，支持多种高级算法，包括多种增强采样方法，复杂的 reaction coordinate restraints, 路径采样(path sampling), 蒙特卡洛采样(Monte Carlo sampling)，路径优化算法及多种分析工具。
除外CHARMM力场，软件还支持使用 AMBER 核酸与蛋白质力场，OPLS 蛋白质力场，Bristol-Myers Squibb 核酸力场。
支持 TIP3P、TIP4P、SPC、SPC/E 及 ST2 等水分子模型。

获得 CHARMM 代码
----------------
CHARMM软件安装包需在官网自行申请 license 进行下载：https://www.charmm.org/charmm/program/obtaining-charmm/

构建 CHARMM 的依赖环境
----------------------

.. code:: shell

   $ srun -p small -n 4 --pty /bin/bash
   $ module load intel-mkl/2020.1.217-intel-19.1.1
   $ module load openmpi/4.0.4-gcc-8.3.0

解压并编译 CHARMM 
-----------------

.. code:: shell

   $ tar -xvf charmm.tar
   $ cd charmm
   $ ./configure
   $ cd build/cmake
   $ make
   $ make install
   $ cd ../../

验证安装是否成功：
-------------------------

.. code:: shell

   $ ./bin/charmm -h

返回信息为：

.. code:: shell

   CHARMM>
   CHARMM> available command line arguments:
   CHARMM> <varname>=<value>  or  <varname>:<value>
   CHARMM>                  sets @ variable in charmm to value
   CHARMM> -h, -help    This text.
   CHARMM> -input, -i f get input script from file f.
   CHARMM> -output,-o f put output to file f.
   CHARMM> -prevclcg    Compatibility mode with previous CLCG.
   CHARMM> -prevrandom  Compatibility mode with previous RANDOM.
   CHARMM> -chsize N    Allocate arrays to hold up to N atoms.
   CHARMM>

参考链接
--------

-  `CHARMM 官网 <https://www.charmm.org/charmm/>`__
