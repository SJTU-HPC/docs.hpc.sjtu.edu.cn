.. _python:

Python
======

本文档向您展示如何使用 Miniconda 在家目录中建立自定义的 Python 环境。不同的 Python 版本 2 或 3，对应不同的 Miniconda。

Miniconda2
----------

加载 Miniconda2

.. code:: bash

   $ module load miniconda2

创建 conda 环境来安装所需 Python 包。

.. code:: bash

   $ conda create --name mypython2 numpy scipy matplotlib ipython jupyter

指定 python 版本（不指定将默认安装最新版）

.. code:: bash

   $ conda create --name mypython2 python==2.7

激活 python 环境

.. code:: bash

   $ source activate mypython2

通过 conda 或 pip 添加更多软件包

.. code:: bash

   $ conda install YOUR_PACKAGE
   $ pip install YOUR_PACKAGE

Miniconda 3
-----------

加载 Miniconda3

.. code:: bash

   $ module load miniconda3

创建conda环境来安装所需Python包。

.. code:: bash

   $ conda create --name mypython3 numpy scipy matplotlib ipython jupyter

激活 python 环境

.. code:: bash

   $ source activate mypython3

通过conda或pip添加更多软件包

.. code:: bash

   $ conda install YOUR_PACKAGE
   $ pip install YOUR_PACKAGE

使用全局预创建的conda环境
-------------------------

π 集群已创建全局的conda环境，该环境主要面向生物医学用户主要包含tensorflow-gpu@2.0.0，R@3.6.1，python@3.7.4
。使用以下指令激活环境：

.. code:: bash

   $ module load miniconda3
   $ source activate /lustre/opt/condaenv/life_sci

conda拓展模块查询方法

.. code:: bash

   $ conda list

R拓展模块查询方法

.. code:: bash

   $ R
   > installed.packages()

使用Miniconda向slurm提交作业
----------------------------

以下为python示例作业脚本，我们将向slurm申请两cpu核心，并在上面通过python打印\ ``hello world``\ 。

.. code:: bash

   #!/bin/bash
   #SBATCH -J hello-python
   #SBATCH -p small
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 2

   module load miniconda3

   source activate mypython3

   python -c "print('hello world')"

我们假定以上脚本内容被写到了\ ``hello_python.slurm``\ 中，使用\ ``sbatch``\ 指令提交作业。

.. code:: bash

   $ sbatch hello_python.slurm

参考资料
--------

-  `miniconda <https://docs.conda.io/en/latest/miniconda.html/>`__
