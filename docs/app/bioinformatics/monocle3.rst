.. _Monocle3:

Monocle 3
==========

简介
----

Monocle 3是一个分析单细胞基因表达数据的工具包。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n monocle3
   source activate monocle3
   conda install r-monocle3

参考资料
--------

-  `Monocle 3 <https://cole-trapnell-lab.github.io/monocle3/docs/installation/>`__
