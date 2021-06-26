.. _Lumpy:

Lumpy
===================

简介
-----------------

lumpy是目前比较流行的一款SV检测工具，它同时支持PEM与SR和RD三种模式。在biostar上很多用户推荐，
在lumpy所发的文章中，与Pindel，delly，gasvpro等软件比较，也有不错的效果。软件使用也非常容易，
不仅支持gemrline样品，也支持somatic样品。

完整步骤
----------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda lumpy-sv
