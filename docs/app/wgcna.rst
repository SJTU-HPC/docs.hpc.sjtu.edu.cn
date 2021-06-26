.. _WGCNA:

WGCNA
=================

简介
------------------

Weighted correlation network analysis, also known as weighted gene co-expression network analysis (WGCNA),
is a widely used data mining method especially for studying biological networks based on pairwise correla
tions between variables.

完整步骤
--------------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda r-wgcna
