.. _DESeq2:

DESeq2 安装
=====================

完整步骤

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bioconductor-deseq2

安装完成后可以在 R 中输入 ``library("DESeq2")`` 检测是否安装成功
