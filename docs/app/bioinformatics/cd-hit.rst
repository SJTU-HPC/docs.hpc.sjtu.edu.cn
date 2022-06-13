.. _cd-hit:

cd-hit
=====================

简介
---------------

cd-hit 是用于蛋白质序列或核酸序列聚类的工具，根据序列的相似度对序列进行聚类以去除冗余的序列，一般用于构建非冗余的数据集。


CPU 版本 cd-hit 安装方法
---------------------------

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n cdhit
   source activate cdhit
   conda install -c bioconda cd-hit

