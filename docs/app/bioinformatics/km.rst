.. _km:

km
==========================

简介
---------------

Knowledge management is the systematic management of an organization's
knowledge assets for creating value and meeting tactical & strategic
requirements. It consists of the initiatives, processes, strategies,
and systems that sustain and enhance the storage, assessment, sharing,
refinement, and creation of knowledge.

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
