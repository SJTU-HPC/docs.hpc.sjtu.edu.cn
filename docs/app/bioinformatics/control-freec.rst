.. _Control-FREEC:

Control-FERRC
=======================

简介
----------------

Control-FREEC is a tool for detection of copy-number changes and allelic imbalances (including LOH)
using deep-sequencing data originally developed by the Bioinformatics Laboratory of Institut Curie
(Paris).

完整步骤
-----------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda control-freec
