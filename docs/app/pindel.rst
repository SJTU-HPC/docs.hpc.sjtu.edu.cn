.. _Pindel:

Pindel
======================================

简介
-----------------

Pindel can detect breakpoints of large deletions, medium sized insertions, inversions,
tandem duplications and other structural variants at single-based resolution from next-
gen sequence data. It uses a pattern growth approach to identify the breakpoints of
these variants from paired-end short reads.

完整步骤
----------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda pindel
