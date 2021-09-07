.. _AUGUSTUS:

AUGUSTUS
=======================


简介
--------------
Augustus  是一个基于 PMML(Predictive Model Markup Language) 的数据统计和挖掘模型

完整步骤
---------------

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c anaconda boost
   conda install -c bioconda augustus
