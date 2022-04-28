.. _DoubletFinder:

DoubletFinder
==============

简介
----

DoubletFinder可用来检测单细胞测序中的Doublet，也就是指双联体细胞。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n doubletfinder
   source activate doubletfinder
   conda install -c paul.martin-2 r-doubletfinder

参考资料
--------

-  `DoubletFinder <https://github.com/chris-mcginnis-ucsf/DoubletFinder>`__
