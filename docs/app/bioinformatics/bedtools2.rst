.. _Bedtools2:

BEDTOOLS2
===================================

简介
-------------------------
Bedtools是一款可以对genomic features进行比较、相关操作和注释的工具，
目前版本已经有三十多个工具／命令用以实现各种不同的功能，可以针对bed、vcf、
gff等格式的文件进行处理。

安装
-------------------------
使用conda安装
^^^^^^^^^^^^^^^
.. code:: bash

    srun -p small -n 4 --pty /bin/bash
    module load miniconda3
    conda create -n mypy #创建自己的环境
    source activate mypy #进入自己的环境
    conda install -c bioconda bedtools

使用git安装
^^^^^^^^^^^^^^^
.. code:: bash
    
    wget https://github.com/arq5x/bedtools2/releases/download/v2.29.1/bedtools-2.29.1.tar.gz
    tar -zxvf bedtools-2.29.1.tar.gz
    cd bedtools2
    make

下载测试数据
^^^^^^^^^^^^^^^
.. code:: bash

    mkdir -p test_data
    cd test_data
    curl -O https://s3.amazonaws.com/bedtools-tutorials/web/maurano.dnaseI.tgz
    curl -O https://s3.amazonaws.com/bedtools-tutorials/web/cpg.bed
    curl -O https://s3.amazonaws.com/bedtools-tutorials/web/exons.bed
    curl -O https://s3.amazonaws.com/bedtools-tutorials/web/gwas.bed
    curl -O https://s3.amazonaws.com/bedtools-tutorials/web/genome.txt
    curl -O https://s3.amazonaws.com/bedtools-tutorials/web/hesc.chromHmm.bed
    tar -zxvf maurano.dnaseI.tgz
    rm maurano.dnaseI.tgz

使用方法与范例
-------------------------
intersect
^^^^^^^^^^^^^^^
bedtools intersect比较两个或多个BED/BAM/VCF/GFF文件，并识别genome中两个文件中的特征重叠的所有区域（即共享至少一个碱基对）。

overlap
""""""""""
.. code:: bash 

    bedtools intersect -a cpg.bed -b exons.bed > result.bed

从cpg.bed中取出与exons.bed不重叠的区域
""""""""""""""""""""""""""""""""""""""""
.. code:: bash 

    bedtools intersect -a cpg.bed -b exons.bed -v > result.bed

多个文件的比较
""""""""""""""""""""
.. code:: bash 

    bedtools intersect -a exons.bed -b cpg.ed gwas.bed hesc.chromHmm.bed -sorted > result.bed

从bam与bed比较
""""""""""""""""""""
.. code:: bash 

    bedtools intersect -abam tmp.bam -b exons.bed > result.bed

指定overlap的最小fraction
""""""""""""""""""""""""""""""
.. code:: bash 

    bedtools intersect -a cpg.bed -b exons.bed -wo -f 0.50

merge
^^^^^^^^^^^^^^^
Bedtools merge 命令可以将重叠的区间或者紧邻的区间合并成一个新的区间。

合并重叠区间形成一个新的区间
""""""""""""""""""""""""""""""
.. code:: bash 

    bedtools merge -i cpg.bed > result_merge.bed


注意事项
-------------------------
- bedtools默认输入文件的分隔符为TAB，除了bam格式的文件；
- 如果未使用-sorted参数，则bedtools默认不支持大于512M的染色体；
- -sorted参数和-g参数必须存在一个；
- 当进行多个文件比较时，染色体的命名方式必须统一，’chrX‘和’X‘不可以同事存在

参考
-------------------------
- bedtools: a powerful toolset for genome arithmetic: https://bedtools.readthedocs.io/en/latest/index.html