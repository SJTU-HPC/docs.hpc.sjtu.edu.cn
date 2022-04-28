.. _STAR-Fusion:

STAR-Fusion
============

简介
----

STAR-Fusion是一款基于STAR比对结果进行融合基因鉴定的软件。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n star-fusion
   source activate star-fusion
   conda install star-fusion

参考资料
--------

-  `STAR-Fusion <https://github.com/STAR-Fusion/STAR-Fusion/wiki>`__
