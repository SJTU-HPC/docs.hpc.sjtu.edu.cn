.. _colmap:

colmap
=========

简介
-------
COLMAP is a general-purpose, end-to-end image-based 3D reconstruction pipeline (i.e., Structure-from-Motion (SfM) and Multi-View Stereo (MVS)) with a graphical and command-line interface. It offers a wide range of features for reconstruction of ordered and unordered image collections. The software runs under Windows, Linux and Mac on regular desktop computers or compute servers/clusters.

COLMAP是一种通用的运动结构 (SfM) 和多视图立体 (MVS) 管道，具有图形和命令行界面。它为有序和无序图像集合的重建提供了广泛的功能。该软件是在新的 BSD 许可下获得许可的。

使用conda在集群上安装colmap
--------------------------------

可以使用以下命令在超算上安装colmap。

.. code:: console
    
    $ srun -p small -n 4 --pty /bin/bash
    $ module load miniconda3
    $ conda create -n env4colmap
    $ source activate env4colmap
    $ conda install -c conda-forge colmap
