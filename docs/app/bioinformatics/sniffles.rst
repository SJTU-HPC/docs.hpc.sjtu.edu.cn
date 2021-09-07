.. _Sniffles:

Sniffles
=======================


简介
------------------

Sniffles is a structural variation caller using third generation sequencing (PacBio or Oxford Nanopore).
It detects all types of SVs (10bp+) using evidence from split-read alignments, high-mismatch regions,
and coverage analysis. Please note the current version of Sniffles requires sorted output from BWA-MEM
(use -M and -x parameter), Minimap2 (sam file with Cigar & MD string) or NGMLR.

完整步骤
------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda sniffles
