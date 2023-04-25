PERCOLATOR
======================

简介
-------------
Percolator是从鸟枪法蛋白质组学数据集中鉴定肽的半监督学习。
Percolator是一种使用半监督机器的算法。学习提高正确和不正确频谱之间的区分标识。 比赛来自搜索诱饵数据库提供分类器的负例，以及来自目标数据库的高分匹配提供正面的例子。 Percolator训练机器学习一种称为支持向量机 (SVM) 的算法来区分通过将权重分配给多个特征。特征示例包括吉祥物得分、前体质量误差、片段质量误差，变量数 修改等。具有最佳权重的特征向量然后用于对来自所有查询的匹配项重新排序，通常会提高敏感性。


使用方法
-----------------
.. code:: bash

    singularity exec /dssg/share/imgs/percolator/percolator_3.05.sif percolator -h

完整步骤
-----------------
.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda percolate
