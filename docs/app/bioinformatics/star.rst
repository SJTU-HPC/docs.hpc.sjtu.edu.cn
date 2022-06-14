.. _star:

STAR
=======

简介
----

STAR是生信领域常用的基因组对比软件，映射速度极快，准确率较高，在RNA-seq领域具有广泛的应用。

可用的版本
----------

+---------+-------+----------+---------------------------------+
| 版本    | 平台  | 构建方式 | 模块名                          |
+=========+=======+==========+=================================+
| 2.7.6a3 | |cpu| | spack    | star/2.7.6a-gcc-11.2.0 思源一号 |
+---------+-------+----------+---------------------------------+
| 2.7.6a3 | |cpu| | spack    | star/2.7.6a-gcc-9.2.0           |
+---------+-------+----------+---------------------------------+

算例下载
---------

.. code:: bash
   
   思源一号:
   mkdir ~/star && cd ~/star
   cp -r /dssg/share/sample/star/* ./    
   gzip -d Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz
   gzip -d Homo_sapiens.GRCh38.86.chr.gtf.gz

   π2.0:
   mkdir ~/star && cd ~/star
   cp -r /lustre/share/sample/star/* ./    
   gzip -d Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz
   gzip -d Homo_sapiens.GRCh38.86.chr.gtf.gz


数据目录如下所示：

.. code:: bash
         
   [hpc@node522 ~]$ tree star/
   star/
   ├── Homo_sapiens.GRCh38.86.chr.gtf.gz
   ├── Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz
   ├── TG_r1.fastq.gz
   └── TG_r2.fastq.gz

0 directories, 4 files

集群上的STAR
----------------

- `一. 思源一号 STAR`_
- `二. π2.0 STAR`_

.. _一. 思源一号 STAR:

一. 思源一号 STAR
--------------------

使用流程如下所示

1.创建基因组索引
~~~~~~~~~~~~~~~~~~

先创建目录

.. code:: bash

   mkdir chr1_index

脚本如下

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=star
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gcc/11.2.0
   module load star/2.7.6a-gcc-11.2.0
   STAR --runThreadN 64 --genomeSAindexNbases 12 --runMode genomeGenerate --genomeDir chr1_index --genomeFastaFiles Homo_sapiens.GRCh38.dna.chromosome.2.fa --sjdbGTFfile Homo_sapiens.GRCh38.86.chr.gtf --sjdbOverhang 99 

2.比对基因组
~~~~~~~~~~~~~~~~~~

脚本如下

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=star
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gcc/11.2.0
   module load star/2.7.6a-gcc-11.2.0
   STAR --runMode alignReads --outSAMtype BAM Unsorted --readFilesCommand zcat --genomeDir chr1_index/ --outFileNamePrefix Homo_sapiens.GRCh38 --readFilesIn TG_r1.fastq.gz TG_r2.fastq.gz

.. _二. π2.0 STAR:

二. π2.0 STAR
--------------------

使用流程如下所示

1.创建基因组索引 π2.0
~~~~~~~~~~~~~~~~~~~~~~~

先创建目录

.. code:: bash

   mkdir chr1_index

脚本如下

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=star
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gcc/9.2.0
   module load star/2.7.6a-gcc-9.2.0
   STAR --runThreadN 40 --genomeSAindexNbases 12 --runMode genomeGenerate --genomeDir chr1_index --genomeFastaFiles Homo_sapiens.GRCh38.dna.chromosome.2.fa --sjdbGTFfile Homo_sapiens.GRCh38.86.chr.gtf --sjdbOverhang 99 

2.比对基因组 π2.0
~~~~~~~~~~~~~~~~~~~

脚本如下

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=star
   #SBATCH --partition=cpu 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load gcc/9.2.0
   module load star/2.7.6a-gcc-9.2.0
   STAR --runMode alignReads --outSAMtype BAM Unsorted --readFilesCommand zcat --genomeDir chr1_index/ --outFileNamePrefix Homo_sapiens.GRCh38 --readFilesIn TG_r1.fastq.gz TG_r2.fastq.gz

运行结果如下所示
-----------------------------------------

1.STAR 思源一号
~~~~~~~~~~~~~~~~~~

对比基因组完成后，会生成以下文件及目录

.. code:: bash

   [hpchgc@node522 ~]$ tree star/
   star/
   ├── 140803.err
   ├── 140803.out
   ├── chr1_index
   │   ├── chrLength.txt
   │   ├── chrNameLength.txt
   │   ├── chrName.txt
   │   ├── chrStart.txt
   │   ├── exonGeTrInfo.tab
   │   ├── exonInfo.tab
   │   ├── geneInfo.tab
   │   ├── Genome
   │   ├── genomeParameters.txt
   │   ├── Log.out
   │   ├── SA
   │   ├── SAindex
   │   ├── sjdbInfo.txt
   │   ├── sjdbList.fromGTF.out.tab
   │   ├── sjdbList.out.tab
   │   └── transcriptInfo.tab
   ├── Homo_sapiens.GRCh38.86.chr.gtf
   ├── Homo_sapiens.GRCh38Aligned.out.bam
   ├── Homo_sapiens.GRCh38.dna.chromosome.2.fa
   ├── Homo_sapiens.GRCh38Log.final.out
   ├── Homo_sapiens.GRCh38Log.out
   ├── Homo_sapiens.GRCh38Log.progress.out
   ├── Homo_sapiens.GRCh38SJ.out.tab
   ├── run.slurm
   ├── TG_r1.fastq.gz
   └── TG_r2.fastq.gz

2.STAR π2.0
~~~~~~~~~~~~~~~~~~

对比基因组完成后，会生成以下文件及目录

.. code:: bash

   [hpc@cas013 data]$ tree  star
   star
   ├── chr1_index
   │   ├── chrLength.txt
   │   ├── chrNameLength.txt
   │   ├── chrName.txt
   │   ├── chrStart.txt
   │   ├── exonGeTrInfo.tab
   │   ├── exonInfo.tab
   │   ├── geneInfo.tab
   │   ├── Genome
   │   ├── genomeParameters.txt
   │   ├── Log.out
   │   ├── SA
   │   ├── SAindex
   │   ├── sjdbInfo.txt
   │   ├── sjdbList.fromGTF.out.tab
   │   ├── sjdbList.out.tab
   │   └── transcriptInfo.tab
   ├── Homo_sapiens.GRCh38.86.chr.gtf
   ├── Homo_sapiens.GRCh38Aligned.out.bam
   ├── Homo_sapiens.GRCh38.dna.chromosome.2.fa
   ├── Homo_sapiens.GRCh38Log.final.out
   ├── Homo_sapiens.GRCh38Log.out
   ├── Homo_sapiens.GRCh38Log.progress.out
   ├── Homo_sapiens.GRCh38SJ.out.tab
   ├── TG_r1.fastq.gz
   └── TG_r2.fastq.gz

参考资料
--------

- STAR官方网站 https://github.com/alexdobin/STAR/
