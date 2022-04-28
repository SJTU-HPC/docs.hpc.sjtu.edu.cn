.. _Harmony:

Harmony
==========

简介
----

Harmony是一个用来整合不同平台的单细胞数据的方法。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n harmony
   source activate harmony
   conda install r-harmony

参考资料
--------

-  `Harmony <https://portals.broadinstitute.org/harmony/articles/quickstart.html>`__
