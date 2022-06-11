.. _bwa:

BWA
======

简介
----

BWA是用于将DNA与大型参考基因组（例如人类基因组）进行比对的开源软件。

可用的版本
------------

+--------+-------+----------+------------------------------------+
| 版本   | 平台  | 构建方式 | 模块名                             |
+========+=======+==========+====================================+
| 0.7.17 | |cpu| | spack    | bwa/0.7.17-intel-2021.4.0 思源一号 |
+--------+-------+----------+------------------------------------+
| 0.7.17 | |cpu| | spack    | bwa/0.7.17-intel-19.0.4            |
+--------+-------+----------+------------------------------------+

算例获取方式
--------------

.. code:: bash

   思源：
   mkdir ~/bwa && cd ~/bwa
   cp -r /dssg/share/sample/bwa/* ./
   gzip -d B17NC_R1.fastq.gz
   gzip -d B17NC_R2.fastq.gz
   
   π2.0:
   mkdir ~/bwa && cd ~/bwa
   cp -r /lustre/share/sample/bwa/* ./
   gzip -d B17NC_R1.fastq.gz
   gzip -d B17NC_R2.fastq.gz

集群上的BWA
--------------------

- `思源一号 BWA`_

- `π2.0 BWA`_

.. _思源一号 BWA:

思源一号上的BWA
-------------------------------------

首先，创建索引文件
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=bwa 
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load bwa
   bwa index -a bwtsw hg19.fa

然后在运行相关数据
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=bwa 
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load bwa
   bwa mem -t 64 hg19.fa B17NC_R1.fastq B17NC_R2.fastq

.. _π2.0 BWA:

π2.0上的BWA
-------------------------------------

创建索引文件
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=bwa 
   #SBATCH --partition=small
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load bwa
   bwa index -a bwtsw hg19.fa

运行相关数据
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=bwa 
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive 
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load bwa
   bwa mem -t 40 hg19.fa B17NC_R1.fastq B17NC_R2.fastq

运行结果
---------

思源一号
~~~~~~~~

.. code:: bash
   
   索引文件创建结果：
   [bwt_gen] Finished constructing BWT in 695 iterations.
   [bwa_index] 1935.82 seconds elapse.
   [bwa_index] Update BWT... 10.76 sec
   [bwa_index] Pack forward-only FASTA... 8.42 sec
   [bwa_index] Construct SA from BWT and Occ... 743.24 sec
   [main] Version: 0.7.17-r1188
   [main] CMD: bwa index -a bwtsw hg19.fa
   [main] Real time: 2750.503 sec; CPU: 2713.697 sec

   64核心运行结果
   [M::mem_process_seqs] Processed 4720686 reads in 456.298 CPU sec, 8.910 real sec
   [main] Version: 0.7.17-r1188
   [main] CMD: bwa mem -t 64 hg19.fa B17NC_R1.fastq B17NC_R2.fastq
   [main] Real time: 184.120 sec; CPU: 7961.600 sec

π2.0
~~~~~~~~

.. code:: bash
   
   索引文件创建结果：
   [bwt_gen] Finished constructing BWT in 695 iterations.
   [bwa_index] 1989.35 seconds elapse.
   [bwa_index] Update BWT... 13.47 sec
   [bwa_index] Pack forward-only FASTA... 13.20 sec
   [bwa_index] Construct SA from BWT and Occ... 739.38 sec
   [main] Version: 0.7.17-r1188
   [main] CMD: bwa index -a bwtsw hg19.fa
   [main] Real time: 2784.274 sec; CPU: 2775.397 sec

   64核心运行结果
   [M::mem_process_seqs] Processed 1520686 reads in 169.987 CPU sec, 4.657 real sec
   [main] Version: 0.7.17-r1188
   [main] CMD: bwa mem -t 40 hg19.fa B17NC_R1.fastq B17NC_R2.fastq
   [main] Real time: 320.462 sec; CPU: 9784.877 sec
  
参考链接：https://github.com/lh3/bwa
