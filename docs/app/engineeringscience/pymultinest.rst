.. _pymultinest:

pymultinest
==================

简介
-------
PyMultiNest library provides programmatic access to MultiNest and PyCuba.


使用conda在集群上安装PyMultiNest
-----------------------------------------

可以使用以下命令在超算上安装PyMultiNest。

.. code:: console
    
    $ srun -p small -n 4 --pty /bin/bash
    $ module load miniconda3
    $ conda create -n env4PyMultiNest -y
    $ source activate env4PyMultiNest
    $ conda install -c conda-forge pymultinest mpi4py -y
