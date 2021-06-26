.. _BICseq2:

BICseq2
=================

简介
-------------
BICseq2 is an algorithm developed for the normalization of  high-throughput
sequencing (HTS) data and detect copy number variations (CNV) in the genome.
BICseq2 can be used for detecting CNVs with or without a control genome.


完整步骤
-----------------
.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bicseq2-norm
