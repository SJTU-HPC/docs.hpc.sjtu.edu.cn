.. _DELLY:

DELLY
=================

简介
-------------

DELLY 的输入是一组 SAM/BAM 格式的对齐 MPS 读取，其中假设每个输入文件是一个单独的库，具有不同的插入大小中位数和标准差。
联合分析所有输入库以实现最佳灵敏度。该方法由两个独立的组件组成，一个双端映射分析组件和一个拆分读取分析组件。

完整步骤
-------------

.. code:: bash

   module load miniconda3
   conda create -n mypy_py27 python=2.7
   source activate mypy_py27
   conda install -c bioconda delly
