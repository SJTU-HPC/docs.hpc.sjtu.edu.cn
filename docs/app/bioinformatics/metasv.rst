.. _MetaSV:

MetaSV
=======================


简介
---------------

Structural variations (SVs) are large genomic rearrangements that vary significantly in
size,making them challenging to detect with the relatively short reads from next-generation
sequencing (NGS).

完整步骤
---------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda metasv
