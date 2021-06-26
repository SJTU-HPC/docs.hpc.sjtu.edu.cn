.. _Manta:

Manta
================

简介
----------------

MANTA is a central hub of all data flows in the organization. Our scanners connect to
various parts of your environment, automatically gather all metadata and reconstruct
complete lineage. You can then visualize the lineage in our native viewer app - on a
level of detail that fits your needs. MANTA is here to help with any task - you can
search throughout the whole lineage, focus on selected parts only, review previous
versions, share your lineage views with others in the organization and a lot more.

完整步骤
--------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda manta
