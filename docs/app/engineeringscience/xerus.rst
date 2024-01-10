.. _xerus:

Xerus  
============

简介
-------
The xerus library is a general purpose library for numerical calculations with higher order tensors, Tensor-Train Decompositions / Matrix Product States and other Tensor Networks.
The focus of development was the simple usability and adaptibility to any setting that requires higher order tensors or decompositions thereof.


使用conda在集群上安装Xerus
--------------------------------------

以在Pi2.0集群安装为例

.. code:: console
    
    $ srun -p cpu -n 8 --pty /bin/bash
    $ module load miniconda3
    $ conda create -n xerus python=3.7
    $ source activate xerus
    $ conda install -c rotekekse -c conda-forge xerus


参考资料
--------

-  `Xerus docs <https://www.libxerus.org/documentation/>`__
-  `Xerus gitlab <https://git.hemio.de/xerus/xerus>`__