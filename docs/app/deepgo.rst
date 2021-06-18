.. _DeepGo:

DeepGo 安装
========================

完整步骤

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/bio-ontology-research-group/deepgo.git
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install pip
   pip install -r requirements.txt
