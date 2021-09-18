.. _km:

km
==========================

简介
---------------

A software for RNA-seq investigation using k-mer decomposition

完整步骤
----------------

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/iric-soft/km.git
   module load miniconda3
   conda create -n mypy
   source activate mypy
   chmod +x easy_install.sh
   ./easy_install.sh
