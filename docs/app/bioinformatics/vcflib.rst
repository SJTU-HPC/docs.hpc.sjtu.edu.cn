.. _vcflib:

vcflib
==========

简介
----

The Variant Call Format (VCF)  是一种平面文件、制表符分隔的文本格式，用于描述个体之间的引用索引变体。 VCF 提供了一种用于描述个体和样本群体变异的通用交换格式，并且已成为各种基因组变异检测器事实上的标准报告格式。

安装方式
----------

可以使用miniconda3进行安装，以Pi2.0为例:

.. code-block:: bash

   srun -p cpu -n 8 --pty /bin/bash
   module load  miniconda3/22.11.1
   conda create -n vcflib python=3.8.16
   conda activate vcflib
   conda install -c bioconda vcflib

参考资料
--------

-  `vcflib <https://github.com/vcflib/vcflib/tree/master>`__
