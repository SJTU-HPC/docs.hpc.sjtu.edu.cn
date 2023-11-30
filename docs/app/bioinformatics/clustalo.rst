.. _Clustalo:

Clustalo
========

简介
----
Clustalo即Clustal-Omega，是一种用于蛋白质和 DNA/RNA 的通用多序列比对 （MSA） 程序。它可以生成高质量的MSA，并能够在合理的时间内处理数十万个序列的数据集。

在默认模式下，用户提供要对齐的序列文件，这些序列被聚类以生成一个指南树，这用于指导序列的“渐进式对齐”。此外，还具有将现有比对相互对齐、将序列与比对以及使用隐马尔可夫模型 （HMM） 来帮助指导与用于制作 HMM 的序列同源的新序列的比对的工具。后一种过程称为“外部轮廓对齐”或 EPA。

Clustal-Omega 使用 HMM 作为对齐引擎，基于 Johannes Soeding 的 HHalign 包。指南树是使用 mBed的增强版本制作的，它可以在 O(N*log(N)) 时间内聚类大量序列。然后，按照参考树给出的聚类，使用 HHalign 对齐越来越大的对齐方式进行多重对齐。

目前形式的 Clustal-Omega 已经过广泛的蛋白质序列测试，自 1.1.0 版起添加了 DNA/RNA 支持。

可用的版本
-----------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 1.2.4  |  cpu    |precompile| clustalo/1.2.4-intel-2021.4.0 思源一号                    |
+--------+---------+----------+-----------------------------------------------------------+
| 1.2.4  |  cpu    |precompile| clustalo/1.2.4-intel-2021.4.0                             |
+--------+---------+----------+-----------------------------------------------------------+


Clustalo预编译文件安装步骤
----------------------------

官网下载预编译文件

.. code:: bash

   $ wget http://www.clustal.org/omega/clustalo-1.2.4-Ubuntu-x86_64

重命名后添加可执行权限

.. code:: bash

   $ mv clustalo-1.2.4-Ubuntu-x86_64 clustalo
   $ chmod u+x clustalo

添加Clustalo的环境变量

.. code:: bash

   $ export PATH=path/to/clustalo:$PATH

检验安装，以下命令查看clustalo版本信息

.. code:: bash

   $ clustalo --version


Clustalo编译安装步骤
---------------------------

安装clustalo之前，要先安装argtable2-13作为其编译的依赖。

.. code:: bash

   $ wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/argtable2/13-1.1/argtable2_13.orig.tar.gz
   $ tar -zxvf argtable2_13.orig.tar
   $ cd argtable2-13/
   $ $ ./configure --prefix=path/to/argtable2-13
   $ make check
   $ make && make install

添加argtable2-13的环境变量

.. code:: bash

   $ export PATH=path/to/argtable2-13:$PATH
   $ export LD_LIBRARY_PATH=path/to/argtable2-13/lib:$LD_LIBRARY_PATH

编译安装clustalo

.. code:: bash

   $ clustalo --version
   $ wget https://launchpad.net/ubuntu/+source/clustalo/1.2.4/clustalo_1.2.4.orig.tar
   $ tar -zxvf clustalo_1.2.4.orig.tar
   $ cd clustal-omega-1.2.4

申请节点进行编译

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash # Pi2.0
   $ srun -p 64c512g -n 4 --pty /bin/bash # 思源一号

调用intel-oneapi编译器

.. code:: bash

   $ module load oneapi/2021.4.0

开始编译

.. code:: bash

   $ ./configure CFLAGS='-I/path/to/argtable2-13/include' LDFLAGS='-L/path/to/argtable2-13/lib' --prefix=/path/to/clustalo
   $ make check
   $ make && make install

添加Clustalo的环境变量

.. code:: bash

   $ export PATH=/path/to/clustalo/bin:$PATH
   $ export PATH=/path/to/clustalo/include:$PATH
   $ export LD_LIBRARY_PATH=/path/to/clustalo/lib:$LD_LIBRARY_PATH

检验安装，以下命令查看clustalo版本信息

.. code:: bash

   $ clustalo --version

Clustalo运行示例
----------------

将以下内容保存为globin.fa

.. code:: bash

   >P01013 GENE X PROTEIN (OVALBUMIN-RELATED)
    QIKDLLVSSSTDLDTTLVLVNAIYFKGMWKTAFNAEDTREMPFHVTKQESKPVQMMCMNNSFNVATLPAE
    KMKILELPFASGDLSMLVLLPDEVSDLERIEKTINFEKLTEWTNPNTMEKRRVKVYLPQMKIEEKYNLTS
    VLMALGMTDLFIPSANLTGISSAESLKISQAVHGAFMELSEDGIEMAGSTGVIEDIKHSPESEQFRADHP
    FLFLIKHNPTNTIVYFGRYWSP

   >NP_689511.2 nuclear autoantigenic sperm protein isoform 3 [Homo sapiens]
   MAMESTATAAVAAELVSADKIEDVPAPSTSADKVESLDVDSEAKKLLGLGQKHLVMGDIPAAVNAFQEAAS
   LLGKKYGETANECGEAFFFYGKSLLELARMENGVLGNALEGVHVEEEEGEKTEDESLVENNDNIDETEGSE
   EDDKENDKTEEMPNDSVLENKSLQENEEEEIGNLELAWDMLDLAKIIFKRQETKEAQLYAAQAHLKLGEVS
   VESENYYOAVEEFQSCLNLQEQYLEAHDRLLAETHYQLGLAYGYNSQYDEAVAQFSKSIEVIENRMAVLNE
   QVKEAEGSSAEYKKEIEELKELLPEIREKIEDAKESQRSGNVAELALKATLVESSTSGFTPGGGGSSVSMI
   ASRKPTDGASSSNCVTDISHLVRKKRKPEEESPRKDDAKKAKQEPEVNGGSGDAVPSGNEVSENMEEEAEN
   QAESRAAVEGTVEAGATVESTAC

module调用clustalo

.. code:: bash

   $ module load clustalo/1.2.4-intel-2021.4.0

Clustal-Omega读取序列文件globin.fa，对齐序列，并将结果以fasta/a2m格式打印到屏幕上。

.. code:: bash

   $ clustalo -i globin.fa

运行结果

.. code:: bash

   >P01013 GENE X PROTEIN (OVALBUMIN-RELATED)
   -------------------QIKDLLVSSS-------TDLD--------------------
   -------TTLV------------LVNAIYFKGM------------WKTAF----------
   --------------------------NAEDTREMPFHVTKQESKPVQMMCMNNSFNVATL
   PAEKMKILELPFASGDLSMLVLLPDEVSDL-------------------------ERIEK
   TINFE-----------------------K-------LTEWTNPNT---------------
   ---------MEKRRVKVYLPQMKIEEKYNLTSVLMALGMTDLFIPSANLTGISSAESLKI
   SQAVHGAFMELSEDGIEMAGS------------------TGVIEDIKH------SPESEQ
   FRADHPFLFLIKHNPT-----NTIVYFGRYWSP---------------------------
   ----
   >NP_689511.2 nuclear autoantigenic sperm protein isoform 3 [Homo sapiens]
   MAMESTATAAVAAELVSADKIEDVPAPSTSADKVESLDVDSEAKKLLGLGQKHLVMGDIP
   AAVNAFQEAASLLGKKYGETANECGEAFFFYGKSLLELARMENGVLGNALEGVHVEEEEG
   EKTEDESLVENNDNIDETEGSEEDDKENDKTEEMPND----------SVLENKSL--QEN
   EEEEIGNLELAWDMLDLAKIIFKRQETKEAQLYAAQAHLKLGEVSVESENYYOAVEEFQS
   CLNLQEQYLEAHDRLLAETHYQLGLAYGYNSQYDEAVAQFSKSIEVIENRMAVLNEQVKE
   AEGSSAEYKKEIEELKELLPEIREKIEDAKE--SQ---------RSGNVA----------
   ELALKATLVESSTSGFTPGGGGSSVSMIASRKPTDGASSSNCVTDISHLVRKKRKPEEES
   PRKDDAKK--AKQEPEVNGGSGDAVPSGNEVSENMEEEAENQAESRAAVEGTVEAGATVE
   STAC

参考资料
--------

-  `Clustal Omega <http://www.clustal.org/omega/>`__
