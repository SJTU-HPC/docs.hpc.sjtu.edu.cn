.. _chimera:

chimera
========================

简介
---------------

This package facilitates the characterisation of fusion products events. It allows to import fusion
data results from the following fusion finders: chimeraScan, bellerophontes, deFuse, FusionFinder,
FusionHunter, mapSplice, tophat-fusion, FusionMap, STAR, Rsubread, fusionCatcher.



完整步骤
-----------------

.. code:: bash

   srun -p small -n 10 --pty /bin/bash
   conda create -n chimer
   source activate chimera
   conda install -c bioconda/label/gcc7 bioconductor-chimera

   以后每次使用时，只需要输入命令: source activate chimera
