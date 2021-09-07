.. _CLEVER:

CLEVER
====================

简介
------------
The clever toolkit (CTK) is a suite of tools to analyze next-generation sequencing data and, in particular, to discover and genotype insertions and deletions from paired-end reads.


完整步骤
------------------
.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda clever-toolkit
