.. _FermiKit:

FermiKit
===================

简介
------------------

FermiKit is a variant calling pipeline for Illumina whole-genome germline data.
It de novo assembles short reads and then maps the assembly against a reference
genome to call SNPs, short insertions/deletions and structural variations.
FermiKit takes about one day to assemble 30-fold human whole-genome data on a
modern 16-core server with 85 GB RAM at the peak, and calls variants in half an
hour to an accuracy comparable to the current practice. FermiKit assembly is a
reduced representation of raw data while retaining most of the original information.



完整步骤
--------------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda fermikit
