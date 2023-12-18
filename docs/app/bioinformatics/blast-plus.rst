.. _blast-plus:

Blast-plus
==========

简介
----
全称Basic Local Alignment Search Tool，即"基于局部比对算法的搜索工具"。
Blast的运行方式是先用目标序列建数据库（这种数据库称为database，里面的每一条序列称为subject），
然后用待查的序列（称为query）在database中搜索，每一条query与database中的每一条subject都要进行双序列比对，从而得出全部比对结果。

blastp：蛋白序列与蛋白库做比对，直接比对蛋白序列的同源性。

blastx：核酸序列对蛋白库的比对，先将核酸序列翻译成蛋白序列（根据相位可以翻译为6种可能的蛋白序列），然后再与蛋白库做比对。

blastn：核酸序列对核酸库的比对，直接比较核酸序列的同源性。

tblastn：蛋白序列对核酸库的比对，将库中的核酸翻译成蛋白序列，然后进行比对。

tblastx：核酸序列对核酸库在蛋白级别的比对，将库和待查序列都翻译成蛋白序列，然后对蛋白序列进行比对。

可用的版本
-----------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 2.9.0  |  arm    |spack     | blast-plus/2.9.0-gcc-9.3.0    ARM                         |
+--------+---------+----------+-----------------------------------------------------------+
| 2.13.0 |  cpu    |precompile| blast-plus/2.13.0-gcc-11.2.0 思源一号                     |
+--------+---------+----------+-----------------------------------------------------------+
| 2.13.0 |  cpu    |precompile| blast-plus/2.13.0-gcc-11.2.0                              |
+--------+---------+----------+-----------------------------------------------------------+

.. _ARM版本BLAST+:


ARM 版本BLAST+
--------------

示例脚本如下(blast.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load blast-plus/2.9.0-gcc-9.3.0
   makeblastdb -in ref.fa -dbtype nucl
   blastn -query in.fa -db ref.fa -out blast_result.txt 
   

使用如下指令提交：

.. code:: bash

   $ sbatch blast.slurm


.. CPU版本BLAST+:

CPU 版本BLAST+
--------------

BLAST+预编译文件安装步骤
----------------------------

官网下载预编译文件

.. code:: bash

   $ wget http://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.13.0/ncbi-blast-2.13.0+-x64-linux.tar.gz

解压

.. code:: bash

   $ tar -zxvf ncbi-blast-2.13.0+-x64-linux.tar.gz

添加BLAST+的环境变量

.. code:: bash

   $ export PATH=path/to/blast/bin:$PATH

检验安装，以下命令查看BLAST+版本信息

.. code:: bash

   $ blastn -version

BLAST+运行示例
----------------

官网下载基因组并解压

.. code:: bash

   $ wget ftp://ftp.ensemblgenomes.org/pub/plants/release-36/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
   $ gzip -d Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz

调用BLAST+

.. code:: bash

   $ module load blast-plus/2.13.0-gcc-11.2.0

构建核酸BLAST数据库

.. code:: bash

   $ makeblastdb -in Arabidopsis_thaliana.TAIR10.dna.toplevel.fa -dbtype nucl -out TAIR10 -parse_seqids

下载拟南芥protein数据

.. code:: bash

   $ wget ftp://ftp.ensemblgenomes.org/pub/plants/release-36/fasta/arabidopsis_thaliana/pep/Arabidopsis_thaliana.TAIR10.pep.all.fa.gz

构建蛋白BLAST数据库

.. code:: bash

   $ gzip -dArabidopsis_thaliana.TAIR10.pep.all.fa.gz
   $ makeblastdb -in  Arabidopsis_thaliana.TAIR10.pep.all.fa -dbtype prot -out TAIR10 -parse_seqids

生成随机序列query.fa

.. code:: bash

   $ echo TGAAAGCAAGAAGAGCGTTTGGTGGTTTCTTAACAAATCATTGCAACTCCACAAGGCGCCTGTAATAGACAGCTTGTGCATGGAACTTGGTCCACAGTGCCCTACCACTGATGATGTTGATATCGGAAAGTGGGTTGCAAAAGCTGTTGATTGTTTGGTGATGACGCTAACAATCAAGCTCCTCTGGT >> query.fa

使用构建好的数据库进行检索

.. code:: bash

   $ blastn -db BLAST/TAIR10 -query query.fa

参考资料
--------

-  `BLAST <https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi>`__
-  `NCBI <https://github.com/ncbi>`__