BOWTIE2
============

简介
--------

Bowtie2 是将测序reads与长参考序列比对工具。适用于将长度大约为50到100或1000字符的reads与相对较长的基因组（如哺乳动物）进行比对。
Bowtie2使用FM索引（基于Burrows-Wheeler Transform 或 BWT）对基因组进行索引，以此来保持其占用较小内存。
对于人类基因组来说，内存占用在3.2G左右。Bowtie2 支持间隔，局部和双端对齐模式。可以同时使用多个处理器来极大的提升比对速度。

π 集群上的 Bowtie2
-----------------------------

查看 π 集群上已编译的软件模块:

.. code:: bash

   $ module avail bowtie2

调用该模块:

.. code:: bash

   $ module load bowtie2/2.3.5.1-intel-19.0.4 

π 集群上的 Slurm 脚本 slurm.test
--------------------------------------------

cpu 队列每个节点配有 40核，这里使用了 1 个节点：

.. code:: bash

   #!/bin/bash

   #SBATCH -J bowtie2_test
   #SBATCH -p cpu
   #SBATCH -n 40
   #SBATCH --ntasks-per-node=40
   #SBATCH -o %j.out
   #SBATCH -e %j.err

   module load bowtie2

   bowtie2-build hsa.fa hsa
   bowtie2 -p 6 -3 5 --local -x hsa -1 example_1.fastq -2 example_2.fastq -S test.sam

π 集群上提交作业
-------------------

.. code:: bash

   $ sbatch slurm.test

参考资料
-------------

-  Bowtie2 : http://bowtie-bio.sourceforge.net/bowtie2/index.shtml
