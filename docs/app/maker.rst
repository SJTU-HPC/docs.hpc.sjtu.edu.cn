.. _MAKER:

MAKER
============================

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda maker
