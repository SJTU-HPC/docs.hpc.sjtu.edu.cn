.. _conda: 

Conda安装软件
===================

本文档介绍使用 Conda 在个人目录中安装生物信息类应用软件。

-  `openslide <#openslide-python>`__
-  `pandas <#pandas>`__
-  `cdsapi <#cdsapi>`__
-  `STRique <#strique>`__
-  `r-rgl <#r-rgl>`__
-  `sra-tools <#sra-tools>`__
-  `DESeq2 <#deseq2>`__
-  `WGCNA <#wgcna>`__
-  `MAKER <#maker>`__
-  `AUGUSTUS <#augustus>`__
-  `DeepGo <#deepgo>`__
-  `km <#km>`__
-  `Requests <#requests>`__

用 Conda 安装软件的流程
-----------------------

加载 Miniconda3

.. code:: bash

   $ module purge
   $ module load miniconda3

创建 conda 环境来安装所需 Python 包（可指定 Python 版本，也可以不指定）

.. code:: bash

   $ conda create --name mypy

激活 python 环境

.. code:: bash

   $ source activate mypy

安装之前，先申请计算节点资源（登陆节点禁止大规模编译安装）

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

通过 conda 安装软件包（有些软件也可以用 pip
安装。软件官网一般给出推荐，用 conda 还是 pip）

.. code:: bash

   $ conda install -c bioconda openslide-python （以 openslide-python 为例）

生信软件在Pi上的使用：用slurm提交作业
-----------------------------------------

Pi 上的计算，需用 slurm 脚本提交作业，或在计算节点提交交互式任务

slurm 脚本示例：申请 small 队列的 2 个核，通过 python 打印
``hello world``

.. code:: bash

   #!/bin/bash
   #SBATCH -J py_test
   #SBATCH -p small
   #SBATCH -n 2
   #SBATCH --ntasks-per-node=2
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module purge
   module load miniconda3

   source activate mypy

   python -c "print('hello world')"

我们假定以上脚本内容被写到了 ``hello_python.slurm`` 中，使用 ``sbatch``
指令提交作业

.. code:: bash

   $ sbatch hello_python.slurm

软件安装示例
------------

许多生信软件可以在 anaconda 的 bioconda package 里找到：

https://anaconda.org/bioconda

以下为一些软件的具体安装步骤：

openslide-python安装
---------------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda openslide-python
   conda install libiconv

pandas安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda pandas

cdsapi安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c conda-forge cdsapi

STRique安装
------------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   git clone --recursive https://github.com/giesselmann/STRique
   cd STRique
   pip install -r requirements.txt
   python setup.py install 

r-rgl安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c r r-rgl

sra-tools安装
--------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda sra-tools

DESeq2安装
-----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bioconductor-deseq2

安装完成后可以在 R 中输入 ``library("DESeq2")`` 检测是否安装成功

WGCNA安装
----------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda r-wgcna

MAKER安装
----------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda maker

AUGUSTUS安装
-------------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda boost
   conda install -c bioconda augustus

DeepGo安装
-----------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/bio-ontology-research-group/deepgo.git
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install pip
   pip install -r requirements.txt

km安装
-------

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/iric-soft/km.git
   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   chmod +x easy_install.sh 
   ./easy_install.sh

Requests安装
-------------

完整步骤

.. code:: bash

   module purge
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda requests

参考资料
--------

-  miniconda https://docs.conda.io/en/latest/miniconda.html
