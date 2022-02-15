.. _AUGUSTUS:

AUGUSTUS
=======================


简介
--------------
Augustus是用作de novo基因注释(即从头预测)的软件。

AUGUSTUS is a gene prediction program for eukaryotes written by Mario Stanke and Oliver Keller. 
It can be used as an ab initio program, which means it bases its prediction purely on the sequence. 
AUGUSTUS may also incorporate hints on the gene structure coming from extrinsic sources such as EST, MS/MS, protein alignments and synthenic genomic alignments.

安装
---------------

.. code:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3
   conda create -n mypy #创建自己的环境
   source activate mypy #进入自己的环境
   conda install -c anaconda boost #安装依赖
   conda install -c bioconda augustus #安装augustus

使用方法与范例
---------------

直接注释
^^^^^^^^^^

若存在已经被训练的物种（augustus --species=help查看），则直接使用代码进行预测基因，以拟南芥（arabidopsis）为例：
 
.. code:: bash
   
   augustus --speices=arabidopsis test.fa > test.gff 

训练注释
^^^^^^^^^^

若不存在被训练过的物种，则需要进行训练

准备训练集与测试集
""""""""""""""""""

根据Augutus的官方教程，应当按如下标准进行基因结构序列准备： 

a. 提供gene的编码部分，包括上游的部分（kb级别）。通常情况下，基因越多（>200），则效果越好，与此同时，外显子的数量也要足够，以便后续训练内含子；

b. 必须保证基因的起始密码子与终止密码子是准确的，并尽量提供较为完整的gene结构和注释信息；

c. 防止gene的冗余，根据Augustus教程的建议，如果任意两个gene在氨基酸水平上的相似度高于70%，那么只需保留一条。这一步既可以避免过度拟合现象，也能用于检验预测的准确性；

d. 多个gene可以在一条序列中，gene可以在正链，亦可在负链，但gene之间不可存在重叠；

e. 每个gene只需要一条转录本，并且以GenBank格式存放

这些注释数据需要随机分为训练集和测试集，为了保证测试集有统计学意义，测试集要足够多的基因（100~200个），并且要足够的随机。

基因结构集的可能来源有:

a. Genbank

b. EST/mRNA-seq的可变剪切联配, 如PASA

c. 临近物种蛋白的可变剪切联配，如GeneWise

d. 相关物种的数据

e. 预测基因的迭代训练

数据集的训练
""""""""""""""""""

.. code:: bash

   # 格式转换；基于选取物种的GFF3以及ref.fa 文件将其转换为Genbank格式
   perl ~/miniconda2/bin/gff2gbSmallDNA.pl ./Spinach_genome/spinach_gene_v1.gff3 ./Spinach_genome/spinach_genome_v1.fa 1000 genes.raw.gb

   # 尝试训练，捕捉错误
   etraining --species=generic --stopCodonExcludedFromCDS=false genes.raw.gb 2> train.err

   # 过滤掉可能错误的基因结构
   cat train.err | perl -pe 's/.*in sequence (\S+): .*/$1/' >badgenes.lst
   filterGenes.pl badgenes.lst genes.raw.gb > genes.gb

   # 提取上一步过滤后的genes.db中的蛋白
   grep '/gene' genes.gb |sort |uniq  |sed 's/\/gene=//g' |sed 's/\"//g' |awk '{print $1}' >geneSet.lst
   python extract_pep.py geneSet.lst Spinach_genome/spinach_pep_v1.fa

   # 将得到的蛋白序列进行建库，自身blastp比对。根据比对结果，如果基因间identity >= 70%，则只保留其中之一，再次得到一个过滤后的gff文件，gene_filter.gff3
   makeblastdb -in geneSet.lst.fa -dbtype prot -parse_seqids -out geneSet.lst.fa
   blastp -db geneSet.lst.fa -query geneSet.lst.fa -out geneSet.lst.fa.blastp -evalue 1e-5 -outfmt 6 -num_threads 8
   python delete_high_identity_gene.py geneSet.lst.fa.blastp Spinach_genome/spinach_gene_v1.gff3

   # 将得到的gene_filter.gff3 转换为genbank 格式文件
   perl ~/miniconda2/bin/gff2gbSmallDNA.pl  gene_filter.gff3  ./Spinach_genome/spinach_genome_v1.fa 1000 genes.gb.filter

   # 将上一步过滤后的文件随机分成两份，测试集和训练集。其中训练集的数目根据gb的LOCUS数目决定，至少要有200（100 为测试集的基因数目，其余为训练集）
   randomSplit.pl genes.gb.filter 100

   # 初始化HMM参数设置（在相应～/minicode/config/species/relative name中形成参数,若之前已经存在该物种名字，则需要删除），并进行训练
   new_species.pl --species=spinach
   etraining --species=spinach genes.gb.filter.train

   # 用测试数据集检验预测效果，这里可以比较我们训练的结果，和近缘已训练物种的训练效果
   augustus --species=spinach genes.gb.filter.test | tee firsttest.out
   augustus --species=arabidopsis genes.gb.filter.test | tee firsttest_ara.out

训练结果的检查
""""""""""""""""""

在 firsttest.out 的尾部可以查看预测结果的统计，首先需要解释几个统计学概念

1. TP(True Positive): 预测为真，事实为真

2. FP(False Positive): 预测为真，事实为假
   
3. FN(False Negative): 预测为假，事实为真
   
4. TN(True Negative): 预测为假，事实为假
   
基于上述，引出下面两个概念：

1. "sensitivity"等于TP/(TP+FP)（预测到的百分率）， 是预测为真且实际为真的占你所有认为是真的比例；
   
2. "specificity"等于TN/(TN+FN)（其中正确的百分率）, 是预测为假且实际为假的占你所有认为是假的比例。我们希望在预测中，尽可能地不要发生误判，也就是没有基因的地方不要找出基因，有基因的地方不要漏掉基因。

重训练以优化结果
""""""""""""""""""
.. code:: bash
   
   # 很有可能的一种情况是，我们第一次的训练结果没有已有训练的效果好，所以我们需要进行循环训练找到最优参数；（运行会非常费时间，而且最终的效果一般只能提高准确度几个百分点，慎重使用）
   optimize_augustus.pl --species=spinach genes.gb.filter.train

   # 再次进行训练，并检验，进行前后比较
   etraining --species=spinach genes.gb.filter.train
   augustus --species=spinach genes.gb.filter.test | tee secondtest.out

   # 如果此时你的gene level的sensitivity还是低于20%说明Trainning set不够大，请添加数据；
   # 如果你获得了满意的Trainning结果，请开始prediction吧

   
氨基酸序列的提取
""""""""""""""""""
.. code:: bash
   
   # 从 firsttest.out 中提取氨基酸序列
   sed -n '/^#/p' firsttest.out | sed -n '/start/,/\]/p' | sed 's/# start gene />/g;s/protein sequence \= \[//g;s/#//g;s/\]//g;s/^\s//g' >seq.fa


参考资料
---------------
- AUGUSTUS official website: http://bioinf.uni-greifswald.de/augustus/

- bioconda augustus：https://anaconda.org/bioconda/augustus

- 使用MAKER进行基因注释(高级篇之AUGUSTUS模型训练）：https://www.jianshu.com/p/679bd6bb0ea4

- Augustus指南（Trainning 部分）：https://www.cnblogs.com/southern-xyx/p/4497497.html

- Augustus Training and Prediction：https://www.cnblogs.com/southern-xyx/p/4497497.html

- Augustus 进行基因注释：https://www.cnblogs.com/zhanmaomao/p/12359964.html
