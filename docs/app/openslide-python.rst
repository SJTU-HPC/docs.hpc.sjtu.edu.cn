.. _openslide-python:

OpenSlide-python
========================

简介
------------

OpenSlide 是一个 C 库，它提供了一个简单的界面来读取整张幻灯片图像（也称为虚拟幻灯片）。当前版本是 3.4.1，发布于 2015-04-20。
Python 和 Java 绑定也可用。Python 绑定包括一个 Deep Zoom生成器和一个简单的基于 Web 的查看器。Java 绑定包括一个简单的图像
查看器。

完整步骤
--------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda openslide-python
   conda install libiconv
