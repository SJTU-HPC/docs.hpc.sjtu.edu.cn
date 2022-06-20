.. _R:

R
==========

简介
----

R 语言是为数学研究工作者设计的一种数学编程语言，主要用于统计分析、绘图、数据挖掘。R 语言是解释运行的语言（与 C 语言的编译运行不同），它的执行速度比 C 语言慢得多，不利于优化。但它在语法层面提供了更加丰富的数据结构操作并且能够十分方便地输出文字和图形信息，所以它广泛应用于数学尤其是统计学领域。



R使用说明
-----------------------------

在思源一号上自行安装并使用R
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 使用conda创建虚拟环境，激活虚拟环境并安装R，然后进入R终端：

.. code:: bash

  srun -p 64c512g -n 4 --pty /bin/bash
  module load miniconda3/4.10.3
  conda create --name R_test
  source activate R_test
  conda install -c conda-forge r-base=4.1.3
  R

2. 在R终端执行R语句，比如：

.. code:: bash

  > print(1+2)
  [1] 3


在pi2.0上自行安装并使用R
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 使用conda创建虚拟环境，激活虚拟环境并安装R，然后进入R终端：

.. code:: bash

  srun -p small -n 4 --pty /bin/bash
  module load miniconda3/4.7.12.1
  conda create --name R_test
  source activate R_test
  conda install -c conda-forge r-base=4.1.3
  R

2. 在R终端执行R语句，比如：

.. code:: bash

  > print(sin(pi/2))
  [1] 1









参考资料
---------

-  `Anaconda 官网 <https://anaconda.org/>`__
-  `如何使用conda安装R和R包 <https://www.jianshu.com/p/b9eb874fc8f4>`__



