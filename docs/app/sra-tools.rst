.. _sra-tools:

sra-tools
===========================

简介
----------------

来自 NCBI 的 SRA 工具包和 SDK 是用于使用 INSDC 序列读取档案中的数据的工具和库的集合。

完整步骤
-----------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda sra-tools
