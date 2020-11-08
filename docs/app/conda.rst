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
