.. _Bismark:

Bismark
=========

简介
----

Bismark可以高效地分析BS-Seq数据，方便地进行读段比对和甲基化探测，Bismark能区分CpG、CHG和CHH，允许用户通过可视化来解释数据。

可用的版本
----------

+-----------+---------+----------+---------------------------------------+
| 版本      | 平台    | 构建方式 | 模块名                                |
+===========+=========+==========+=======================================+
| 0.23.0    | |cpu|   | Spack    | `bismark/0.23.0-gcc-11.2.0`_ 思源一号 |
+-----------+---------+----------+---------------------------------------+
| 0.19.0    | |cpu|   | Spack    | `bismark/0.19.0-intel-19.0.4`_        |
+-----------+---------+----------+---------------------------------------+

如何使用
---------

.. _bismark/0.23.0-gcc-11.2.0:

思源一号集群 Bismark
^^^^^^^^^^^^^^^^^^^^^^

在思源一号集群上使用如下命令:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load bismark/0.23.0-gcc-11.2.0
   bismark --help

.. _bismark/0.19.0-intel-19.0.4:

π 集群 Bismark
^^^^^^^^^^^^^^^^^

在 π 集群上使用如下命令:    

.. code-block:: bash

   srun -p small -n 4 --pty /bin/bash
   module load bismark/0.19.0-intel-19.0.4
   bismark --help

使用Conda安装及运行
--------------------

使用 Conda 安装 Bismark
^^^^^^^^^^^^^^^^^^^^^^^^^

推荐使用 ``Conda`` 在用户目录部署特定的 ``Bismark`` 软件，以思源一号为例：

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda create -n biotools                 # 创建新的环境
   source activate biotools                 # 激活环境
   conda install -c bioconda bismark=0.23.1 bowtie2=2.2.1 # 安装bismark
   bismark --help
   
基因组
^^^^^^^

.. code-block:: bash

   wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz
   gunzip -c hg19.fa.gz > ~/hg19/hg19.fa     # 需要创建~/hg19目录

运行示例
^^^^^^^^^^^

建立基因组索引
""""""""""""""""""

.. code-block:: bash

   bismark_genome_preparation ~/hg19

序列比对
"""""""""""

.. code-block:: bash

   cp $(which test_data.fastq) .
   bismark --genome ~/hg19 test_data.fastq
   # 输出
   ├── test_data_bismark_bt2.bam
   └── test_data_bismark_bt2_SE_report.txt

查看BAM文件 ``samtools view test_data_bismark_bt2.bam | head -n5`` ：

.. code-block:: bash

   SRR020138.15024317_SALK_2029:7:100:1672:902_length=86	16	chr1	57798677	42	50M	*	0	0	TTCTTTCCCATCCCATAAATCCTAAAAATAATAAAAAATCATCCCCAAAT	@@:AC@<=+@?+8)@BCCCA=6BCCCCCCCCCCCCCCCCACB=<88BCCA	NM:i:11	MD:Z:14G2G6G0G0G0G4G1G1G0G10G1	XM:Z:..............z..h......hhhh....h.h.hh..........h.	XR:Z:CT	XG:Z:GA
   SRR020138.15024318_SALK_2029:7:100:1672:137_length=86	0	chr12	129774096	8	50M	*	0	0	AAAAAAAAAAAAAAGAAAAAAAAGAAAAAGAAAAGGAAAAGTAAAAAAAA	=@CAA=@B@CB=98%:AB?>@56/=3<=<)>B@:*=:=61%,<A@@1+12	NM:i:2	MD:Z:41C5G2	XM:Z:.........................................h........	XR:Z:CT	XG:Z:CT
   SRR020138.15024319_SALK_2029:7:100:1672:31_length=86	0	chr2	10166575	42	50M	*	0	0	ATTTTGTTATAGAGTGGGGTATTTTCGGGAAGAAGGAGGAGGAGTGTATT	BCCCCBCCCCA?:=ACCBCABCCCCCBCCA??5=9@4BB@;??B@BABBA	NM:i:8	MD:Z:1C1C5C9C2C0C22C1C1	XM:Z:.h.x.....x.........h..hh.Z....................h.x.	XR:Z:CT	XG:Z:CT
   SRR020138.15024320_SALK_2029:7:100:1672:1164_length=86	16	chr5	28344472	8	50M	*	0	0	CACAAAATATCAACACCCCTAAACCCCACATTATTCAAAAATCAATTATA	@@@BBBA@A9=A@<?::2:<CB@?=:BBAC??CB@@BBBBC>:ACABCAB	NM:i:11	MD:Z:4G1G1G3G9G9G5G0G0G1T4G2	XM:Z:....x.h.h...x.........h.........h.....hhh......h..	XR:Z:CT	XG:Z:GA
   SRR020138.15024321_SALK_2029:7:100:1672:433_length=86	0	chr14	38711099	42	50M	*	0	0	TTTTGAGTAGAGAAGTTAGTATTTTAGGGAATTTTTGATTTTTTTAAGTT	BCCBB?B@@A>@-4BBB:7@BBBCBBC@@=A@BCACA;BCBBCBB@@@BB	NM:i:14	MD:Z:0C0C13C0C6C0C6C0C1C3C0C1C0C1C5	XM:Z:hh.............hx......hx......hh.x...hh.hh.h.....	XR:Z:CT	XG:Z:CT

甲基化call字符串对于BS-read中不涉及胞嘧啶的每个位置都用一个点 ``.`` 代替，或包含以下不同的胞嘧啶甲基化的字母 `(大写=甲基化，小写=未甲基化)` ：

.. code-block:: bash

   X # 代表CHG中甲基化的C
   x # 代表CHG中非甲基化的C
   H # 代表CHH中甲基化的C
   h # 代表CHH中非甲基化的C
   Z # 代表CpG中甲基化的C
   z # 代表CpG中非甲基化的C
   U # 代表其他情况的甲基化C(CN或者CHN)
   u # 代表其他情况的非甲基化C (CN或者CHN)
   . # 该位置不是胞嘧啶

去除重复
"""""""""""

.. code-block:: bash

   deduplicate_bismark --bam test_data_bismark_bt2.bam
   # 输出
   ├── test_data_bismark_bt2.deduplicated.bam
   └── test_data_bismark_bt2.deduplication_report.txt

提取甲基化水平
""""""""""""""""""

默认情况下，软件会自动根据 `甲基化的C的类型 (CpG, CHG, CHH)` 和 `比对到四条链上 (OT, OB, CTOT, CTOB)` 两个因素生成结果文件。

- OT -- original top strand
- CTOT -- complementary to original top strand
- OB -- original bottom strand
- CTOB -- complementary to original bottom strand

.. code-block:: bash

   # extract context-dependent (CpG/CHG/CHH) methylation
   cpanm GD::Graph::lines                   # 安装画图的依赖模块，非必须
   bismark_methylation_extractor --gzip --bedGraph test_data_bismark_bt2.deduplicated.bam
   # 输出
   ├── CHG_OB_test_data_bismark_bt2.deduplicated.txt.gz
   ├── CHG_OT_test_data_bismark_bt2.deduplicated.txt.gz
   ├── CHH_OB_test_data_bismark_bt2.deduplicated.txt.gz
   ├── CHH_OT_test_data_bismark_bt2.deduplicated.txt.gz
   ├── CpG_OB_test_data_bismark_bt2.deduplicated.txt.gz
   ├── CpG_OT_test_data_bismark_bt2.deduplicated.txt.gz
   ├── test_data_bismark_bt2.deduplicated.bedGraph.gz
   ├── test_data_bismark_bt2.deduplicated.bismark.cov.gz
   ├── test_data_bismark_bt2.deduplicated.M-bias_R1.png
   ├── test_data_bismark_bt2.deduplicated.M-bias.txt
   └── test_data_bismark_bt2.deduplicated_splitting_report.txt

查看结果 ``zcat CHG_OB_test_data_bismark_bt2.deduplicated.txt.gz | head -n5`` ：

.. code-block:: bash
   
   # Bottom链在CHG背景下的甲基化信息
   SRR020138.15024320_SALK_2029:7:100:1672:1164_length=86	-	chr5	28344484	x
   SRR020138.15024320_SALK_2029:7:100:1672:1164_length=86	-	chr5	28344476	x
   SRR020138.15024326_SALK_2029:7:100:1672:1418_length=86	-	chr5	126218386	x
   SRR020138.15024326_SALK_2029:7:100:1672:1418_length=86	-	chr5	126218354	x

网页报告
""""""""""""""""""

.. code-block:: bash

   bismark2report
   # 输出
   └── test_data_bismark_bt2_SE_report.html

参考资料
--------

-  `Bismark 文档 <https://github.com/FelixKrueger/Bismark/tree/master/Docs>`__
