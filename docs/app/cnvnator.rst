.. _CNVnator:

CNVnator
=====================

简介
------------

一种用于发现和表征群体基因组测序数据中拷贝数变异 (CNV) 的工具。该工具也非常适合个人基因组分析。
该方法基于均值偏移、多带宽分区和 GC 校正。

完整步骤
-------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda -c conda-forge cnvnator
