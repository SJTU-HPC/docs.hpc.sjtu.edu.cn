.. _CNVkit:

CNVkit
=======

简介
----

CNVkit是一个Python库和命令行软件工具包，用于研究CNV（Copy number variation）拷贝数变异的软件。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n cnvkit
   source activate cnvkit
   conda install cnvkit

参考资料
--------

-  `CNVkit 文档 <https://github.com/etal/cnvkit>`__
