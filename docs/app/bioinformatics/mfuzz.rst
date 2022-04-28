.. _Mfuzz:

Mfuzz
==========

简介
----

Mfuzz是一个时间序列表达模式聚类分析的R包。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n mfuzz
   source activate mfuzz
   conda install bioconductor-mfuzz

参考资料
--------

-  `Mfuzz <https://rdrr.io/bioc/Mfuzz/>`__
