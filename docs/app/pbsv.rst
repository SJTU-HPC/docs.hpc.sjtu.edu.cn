.. _pbsv:

PBSV
========================

简介
-------------

pbsv is a suite of tools to call and analyze structural variants in diploid genomes from PacBio
single molecule real-time sequencing (SMRT) reads. The tools power the Structural Variant Calling
analysis workflow in PacBio's SMRT Link GUI. pbsv calls insertions, deletions, inversions, duplic
ations, and translocations. Both single-sample calling and joint (multi-sample) calling are provided.

完整步骤
---------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda pbsv
