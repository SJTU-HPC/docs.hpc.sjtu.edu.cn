.. _Manta:

Manta
======

简介
----
Manta软件可以从比对文件中检测SVs和indels。它主要开发用于检测单个样品的germline变异和tumor/normal配对样品的somatic变异。
Manta通过连续组装的方法可以使分辨率达到碱基级别，更有利于下游的注释和临床意义分析。Manta软件接受输入BAM或CRAM格式文件，
并以VCF4.1的格式报告所有的SV和indels突变。

.. _ARM版本Manta:


ARM 版本Manta
-------------

示例脚本如下(manta.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module use /lustre/share/spack/modules/kunpeng920/linux-centos7-aarch64
   module load manta/1.6.0-gcc-9.3.0
   configManta.py --bam test.bam --referenceFasta hg19.fa --runDir YOUR_MANTA_ANALYSIS_PATH
   

使用如下指令提交：

.. code:: bash

   $ sbatch manta.slurm


