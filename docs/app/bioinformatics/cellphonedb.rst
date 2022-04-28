.. _CellPhoneDB:

CellPhoneDB
============

简介
----

CellPhoneDB是公开的人工校正的，储存受体、配体以及两种相互作用的数据库。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n cpdb
   source activate cpdb
   conda install python=3.7 r=4.1
   pip install cellphonedb

参考资料
--------

-  `CellPhoneDB <https://pypi.org/project/CellPhoneDB/>`__
