.. _CREST:

CREST
====================

简介
--------------

CREST (Clipping Reveals Structure) is a new algorithm for detecting genomic structural variations at base-pair resolution using next-generation sequencing data

完整步骤
--------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda blat
   conda install -c bioconda cap3
   conda install -c bioconda samtools
   conda install -c bioconda perl-bioperl
   conda install -c bioconda perl-bio-db-sam
   conda install -c imperial-college-research-computing crest
