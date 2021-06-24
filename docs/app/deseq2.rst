.. _DESeq2:

DESeq2
=====================

简介
---------------

分析来自 RNA-seq 的计数数据的一项基本任务是检测差异表达的基因。计数数据以表格的形式呈现，其中报告了每个样本已分配给每
个基因的序列片段的数量。其他检测类型也有类似的数据，包括比较 ChIP-Seq、HiC、shRNA 筛选和质谱分析。一个重要的分析问
题是与条件内的变异性相比，条件之间的系统变化的量化和统计推断。DESeq2 包提供了使用负二项式广义线性模型测试差异表达的方
法；离散度和对数倍数变化的估计包含数据驱动的先验分布。此小插图解释了包的使用并演示了典型的工作流程。Bioconductor 网
站上的RNA-seq 工作流程涵盖了与此小插图类似的材料，但速度较慢，包括从 FASTQ 文件生成计数矩阵。


完整步骤
-------------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bioconductor-deseq2

安装完成后可以在 R 中输入 ``library("DESeq2")`` 检测是否安装成功
