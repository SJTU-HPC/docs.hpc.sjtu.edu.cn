.. _celldex:

celldex
==========

简介
----

celldex包提供一组参考表达数据集，这些数据集带有精确的细胞类型标签，用于单细胞数据的自动注释或批量RNA序列的反卷积等过程。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n celldex
   source activate celldex
   conda install bioconductor-celldex

参考资料
--------

-  `LTLA/celldex <https://rdrr.io/github/LTLA/celldex/>`__
