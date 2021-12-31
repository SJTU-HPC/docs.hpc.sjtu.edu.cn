.. _chimera:

chimera
========================

简介
---------------

This package facilitates the characterisation of fusion products events. 
It allows to import fusiondata results from the following fusion finders:
chimeraScan, bellerophontes, deFuse, FusionFinder,FusionHunter, mapSplice,
tophat-fusion, FusionMap, STAR, Rsubread, fusionCatcher.

完整步骤
-----------------

.. code:: bash

   srun -p small -n 10 --pty /bin/bash 
   conda create -n chimera
   source activate chimera
   conda install -c bioconda/label/gcc7 bioconductor-chimera
   source activate chimera
