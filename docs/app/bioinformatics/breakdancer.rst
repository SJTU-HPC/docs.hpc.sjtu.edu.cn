.. _BreakDancer:

BreakDancer
======================

简介
--------------
BreakDancer包含两个互补的程序：BreakDancerMax和BreakDancerMini。
BreakDancerMax根据二代测序read比对时，出现的异常比对，预测插入，缺失，倒位，染色体间或染色体内易位等五种结构突变。

完整步骤
------------------
.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda breakdancer
