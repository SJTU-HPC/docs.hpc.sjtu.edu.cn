.. _BASIL-ANISE:

BASIL-ANISE
======================

简介
-------------
BASIL 是一种从 BAM 格式的对齐配对 HTS 读数中检测结构变体（包括插入断点）断点的方法。

完整步骤
-----------------
.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda anise_basil
