.. _BICseq2:

BICseq2
==========


简介
--------------------

BICseq2 是一种为高通量测序 (HTS) 数据标准化
和基因组中的拷贝数变异 (CNV) 检测
而开发的算法。
无论是否存在对照基因组，BICseq2 都可以进行CNV的检测。

   BICseq2 is an algorithm developed for the normalization of  high-throughput
   sequencing (HTS) data and detect copy number variations (CNV) in the genome.
   BICseq2 can be used for detecting CNVs with or without a control genome.

下载与安装
--------------------

conda安装
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda bicseq2-norm # 对测序数据进行潜在偏差修正
   conda install -c bioconda bicseq2-seg # 可以对bicseq2-norm的结果进行CNV检测

使用与范例
--------------------

BICseq2-norm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用方法
""""""""""""""""""""

.. code:: bash
   
   BICseq2-norm.pl [options] <configFile> <output>
  
- configFile: 配置文件，包含规范化所需的信息。制表符分割，内容说明见下方表格: 
 
   +--------------------+----------------------------------------+
   |     column name    |          description                   |
   +====================+========================================+
   |     chromName      |       chromosome name                  |
   +--------------------+----------------------------------------+
   |     faFile         | reference sequence of this chromosome  |
   |                    | (human hg18, hg19 ...)                 |
   +--------------------+----------------------------------------+
   |     MapFile        |   mappability file of this chromosome  |
   +--------------------+----------------------------------------+
   |     readPosFile    | all the mapping positions of all reads |
   |                    | that uniquely mapped to this chromosome|
   +--------------------+----------------------------------------+
   |     binFile        | normalized data. The data will be      |
   |                    | binned with the bin size as specified  |
   |                    | by the option -b                       | 
   +--------------------+----------------------------------------+

- output: 用于存储GAM model中的参数估计，一般用户无需考虑

参数解释
""""""""""""""""""""

  - --help
  - -l=<int>: read length
  - -s=<int>: fragment size
  - -p=<float>: a subsample percentage: default 0.0002.
  - -b=<int>: bin the expected and observed as <int> bp bins; Default 100.
  - --gc_bin: if specified, report the GC-content in the bins
  - --NoMapBin: if specified, do NOT bin the reads according to the mappability
  - --bin_only: only bin the reads without normalization
  - --fig=<string>: plot the read count VS GC figure in the specified file (in pdf format)
  - --title=<string>: title of the figure
  - --tmp=<string>: the tmp directory


BICseq2-seg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用方法
""""""""""""""""""""

BICseq2-seg.pl [options] <configFile> <output>

- configFile: 存储检测CNV过程中所需的信息，内容详解见下方表格

  - 没有control genome的情况
     
   +--------------------+----------------------------------------+
   |     column name    |          description                   |
   +====================+========================================+
   |     chromName      |       chromosome name                  |
   +--------------------+----------------------------------------+
   |     binFileNorm    | normalized bin file as obtained        |
   |                    | from BICseq2-norm                      |
   +--------------------+----------------------------------------+

  - 有control genome的情况 
   
   +--------------------+----------------------------------------+
   |     column name    |          description                   |
   +====================+========================================+
   |     chromName      |       chromosome name                  |
   +--------------------+----------------------------------------+
   |   binFileNorm.Case | normalized bin file of the case genome |
   |                    | as obtained from BICseq2-norm          |
   +--------------------+----------------------------------------+
   | binFileNorm.Control| normalized bin file of the control     |
   |                    | genome as obtained from BICseq2-norm   |
   +--------------------+----------------------------------------+

- output: 结果文件

参数解释
""""""""""""""""""""

  - --lambda=<float>: the (positive) penalty used for BICseq2
  - --tmp=<string>: the tmp directory
  - --help: pring this message
  - --fig=<string>: plot the CNV profile in a png file
  - --title=<string>: the title of the figure
  - --nrm: do not remove likely germline CNVs (with a matched normal) or segments with bad mappability (without a matched normal)
  - --bootstrap: perform bootstrap test to assign confidence (only for one sample case)
  - --noscale: do not automatically adjust the lambda parameter according to the noise level in the data
  - --strict: if specified, use a more stringent method to ajust the lambda parameter
  - --control: the data has a control genome
  - --detail: if specified, print the detailed segmentation result (for multiSample only)

参考连接
--------------------

- BIC-seq: a fast algorithm for detection of copy number alterations based on high-throughput sequencing data: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3026225/
- github: https://github.com/ding-lab/BICSEQ2
- http://compbio.med.harvard.edu/BIC-seq/