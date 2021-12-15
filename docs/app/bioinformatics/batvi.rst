.. _BatVI:

BatVI
=====================

简介
--------------
检测病毒整合的软件包

安装
----------------

通过conda安装
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: bash

   module load miniconda3
   conda create -n mypy #创建环境
   source activate mypy #进入环境
   conda install -c bioconda batvi #安装BatVI

通过链接下载
^^^^^^^^^^^^^^^^^^^^^^^^
https://www.comp.nus.edu.sg/~bioinfo/batvi/batvi1.03.tar.gz

.. code:: bash
   
   #解压
   tar -zxvf batvi1.00.tar.gz
   cd batvi1.00
   ./build.sh

使用与范例
----------------

CONFIGURATION FILE 
^^^^^^^^^^^^^^^^^^^^^^^^
在BatVI目录下，创建名为 batviconfig.txt 的文件，文件中的内容如下: 

.. code:: bash 
   
   INDEX=< path_to_pathogen_batmis_index >
   PATHOGEN_BLAST_DB=< path_to_pathogen_blast_index >
   HG_BLAST_DB=< path_to_human_blast_index >
   HG_GENOME=< compressed_human_genome >
   HG_BWA=< path_to_human_BWA_index >
   PATHOGEN_BWA=< path_to_pathogen_BWA_index >
   BLAST_PATH=< path_to_blast_binary >
   BWA_PATH=< path_to_BWA_binary >
   PICARD_PATH=< path_to_picard_jarfiles >
   SAMTOOLS_PATH=< path_to_samtools_binary >
   BEDTOOLS_PATH=< path_to_bedtools_binary >

batviconfig.txt内容详解如下（可以通过gen_path.sh尝试自动获取路径）: 

.. code:: bash 
   
   INDEX is the batmis index of the virus database.
   HG_BLAST_DB is the blast index of the virus database.
   PATHOGEN_BLAST_DB is the blast index of the human genome.
   HG_GENOME contains a compressed version of the human genome.
   HG_BWA is the BWA index of human.
   PATHOGEN_BWA is the BWA index of the virus database.
   BLAST_PATH is the path to the blast binary
   BWA_PATH is the path to the BWA binary
   PICARD_PATH is the path to the picard jarfiles
   SAMTOOLS_PATH is the path to the samtools binary
   BEDTOOLS_PATH is the path to the bedtools binary

Example of batviconfig.txt

.. code:: bash 

   #This is an example batviconfig.txt file.
   #Comments can be written preceded by a hash..
   INDEX=/mnt/projects/HBVall/batmis/HBVall.fa
   PATHOGEN_BLAST_DB=/mnt/projects/blast/HBVall/HBVall.fa
   HG_BLAST_DB=/mnt/projects/blast/hg19/hg19.fa
   HG_GENOME=/mnt/projects/hg19/hg19.fa
   HG_BWA=/mnt/projects/bwa/hg19/hg19.fa
   PATHOGEN_BWA=/mnt/projects/bwa/HBVall/HBVall.fa
   BLAST_PATH=/usr/bin
   BWA_PATH=/usr/bin
   PICARD_PATH=/usr/bin/picard/
   SAMTOOLS_PATH=/usr/bin
   BEDTOOLS_PATH=/usr/bin

RUNNING THE PROGRAM 
^^^^^^^^^^^^^^^^^^^^^^^^

准备输入文件
""""""""""""""""""
1. pair-ended fastq files
   
   .. code:: bash 

      A_1.fq
      A_2.fq
      B_1.fq
      B_2.fq
   
2. filelist.txt
   
   .. code:: bash

      A_1.fq;A_2.fq;800
      B_1.fq;B_2.fq;800

病毒整合脚本
""""""""""""""""""

.. code:: bash 

   call_integratons.sh <processing_directory> [ options ]

   #processing_directory必须包含filelist.txt，可以包含batviconfig.txt
   #options选项
   #-l|--log  - Name of the log files to write processing information to
   #-t|--threads  - Number of threads to use
   #-f|--filterdup - Filter out duplicate reads

OUTPUT FILES AND FORMAT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

final_hits.txt
""""""""""""""""""
查看输出文件 final_hits.txt 以确认最佳选择，文件内容详解如下: 

.. code:: bash 

   LIB : Library name
   Chr : Chromosome of the human integration
   Human Pos : Location of the human integration
   Sign : Orientation of the human integration
   Viral Sign : Orientation of the viral integration
   Viral Pos : Location of the viral integration
   Read Count : Number of reads used in the prediction of integration. If the entry is marked MSA, the integration has been found using assembly.
   Split Reads : Number of split reads involved in the predicion of the integration. A higher number indicreases the confidence of a prediction.
   Uniquely Mapped Reads : Number of unique mappings to the human genome involved in the prediction. A higher number increases the confidence of a prediction.
   Multiply Mapped Reads : Number of multiple mappings used in predicting the integration.
   Rank1 Hits : Number of reads which have the tophit near the prediction. A higher number increases the confidence of a prediction. The last two columns of this file are to be ignored for this version of BatVI.

predictions.opt.subopt.txt
""""""""""""""""""""""""""""""""""""
在文件 predictions.opt.subopt.txt 中查看所有候选位点，predictions.opt.subopt.txt详解如下: 

.. code:: bash 

   Sign : Orientation of the human integration
   Chr : Chromosome of the human integration
   Human St : Location of the human integration
   Human Ed : Location of the end of the human integration found by reads
   Viral Sign : Orientation of the viral integration
   Viral St : Location of the viral integration
   Viral Ed : Location of the end of the viral integration found by reads
   Median_Rank : Median rank of the prediction
   Read Count : Number of reads used in the prediction of integration.
   Split Reads : Number of split reads involved in the predicion of the integration.
   Uniquely Mapped Reads : Number of unique mappings to the human genome involved in the prediction.
   Multiply Mapped Reads : Number of multiple mappings used in predicting the integration.
   Rank1 Hits : Number of reads which have the tophit near the prediction. The other columns of this file are to be ignored for this version of BatVI.


XXX.predictionsx.msa.txt
""""""""""""""""""""""""""""""""""""
在 tmp.batvi/XXX.predictionsx.msa.txt 中查看（XXX即fastq文件所在目录的名称），文件详解如下: 

.. code:: bash 

   Chr : Chromosome of the human integration
   Human Pos : Location of the human integration
   Sign : Orientation of the human integration
   Viral Pos : Location of the viral integration
   Viral Sign : Orientation of the viral integration
   Integration type : If the entry is marked TOPHIT, the assembly maps to this location as the best hit. If the entry is marked REPEAT, the assembly can map to at least one other location with similar confidence, and is therefore ambiguous.

范例下载
^^^^^^^^^

Example数据下载可使用网址: biogpu.ddns.comp.nus.edu.sg/~ksung/batvi/test

|  There is a set of test fastq files in the test directory (biogpu.ddns.comp.nus.edu.sg/~ksung/batvi/test). 
   The filelist.txt file is already present. 
   To run a test, a HBV database fasta file is given in HBVall.fa. 
   You can download the hg19 fasta file from UCSC genome browser. 
   After you run the program on this data set, the expected output is given in expected.txt.

注意事项
^^^^^^^^^

- Please ensure that the white spaces in the fasta files and genome names are removed or replaces with a character like an underscore.

- The index builder bwtformatdb might crash with a message like "Building cached SA index... xxxx/build_index: 47 line: 168087 Segmentation fault (core dump) ....". This is OK as the required indexes would have already been built at this stage.
  
- The installation can be cleaned using the command 'build.sh clean '.
  
- A directory can be prepared for a fresh run of BatVI with the command 'clean_run.sh '.

参考
----------------

- BatVI official website: https://www.comp.nus.edu.sg/~bioinfo/batvi/index.html
  
- Tennakoon C, Sung WK. BATVI: Fast, sensitive and accurate detection of virus integrations. BMC Bioinformatics. 2017;18(Suppl 3):71. Published 2017 Mar 14. doi:10.1186/s12859-017-1470-x
