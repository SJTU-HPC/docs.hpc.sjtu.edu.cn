.. _cdsapi:

cdsapi
=============================

简介
--------------
The Climate Data Store (CDS) API is the service to allow users to request
data from CDS datasets via a python script. These scripts use a number of
keywords which vary from dataset to dataset, usually following the sections
of the CDS download form.As the CDS API cannot currently return the valid
keyword list on demand, they are documented on this page for some of the
most popular CDS datasets.

完整步骤
-----------------
.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c conda-forge cdsapi
