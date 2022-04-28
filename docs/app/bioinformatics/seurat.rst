.. _Seurat:

Seurat
==========

简介
----

Seurat是由 *New York Genome Center, Satija Lab* 开发的单细胞数据分析集成软件包。其功能不仅包含基本的数据分析流程，如质控，细胞筛选，细胞类型鉴定，特征基因选择，差异表达分析，数据可视化等。同时也包括一些高级功能，如时序单细胞数据分析，不同组学单细胞数据整合分析等。

安装方式
----------

可以使用Conda进行安装，以思源一号为例:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda create -n seurat
   source activate seurat
   conda install r-seurat

参考资料
--------

-  `Seurat <https://satijalab.org/seurat/>`__
