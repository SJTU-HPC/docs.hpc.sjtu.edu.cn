.. _CREST:

CREST
====================

简介
--------------

Crest was introduced in the United States as "Fluoristan" in 1954, as it contained stannous fluoride.
In 1955, the name of the product was changed to "Crest with Fluoristan." The composition of the tooth
-paste had been developed by Joseph C. Muhler, Harry Day, and William H.

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
