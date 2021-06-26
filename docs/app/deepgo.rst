.. _DeepGo:

DeepGo
========================

简介
---------------

使用深度学习识别分类器根据序列和相互作用预测蛋白质功能。

完整步骤
--------------------

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   git clone https://github.com/bio-ontology-research-group/deepgo.git
   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install pip
   pip install -r requirements.txt
