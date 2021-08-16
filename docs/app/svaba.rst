.. _SvABA:

SvABA
=================

简介
------------

SvABA is a method for detecting structural variants in sequencing data using genome-wide local assembly.
Under the hood, SvABA uses a custom implementation of SGA (String Graph Assembler) by Jared Simpson,
and BWA-MEM by Heng Li. Contigs are assembled for every 25kb window (with some small overlap) for every
region in the genome. The default is to use only clipped, discordant, unmapped and indel reads, although
this can be customized to any set of reads at the command line using VariantBam rules. These contigs are
then immediately aligned to the reference with BWA-MEM and parsed to identify variants. Sequencing reads
are then realigned to the contigs with BWA-MEM, and variants are scored by their read support.

使用module运行svaba
---------------------

使用sbatch提交运行脚本(svaba.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=svaba
   #SBATCH --partition=small   
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load svaba/1.1.3-gcc-4.8.5
   svaba run.sh

脚本run.sh示例如下(svaba.slurm、run.sh和数据要在同一目录下):
   
.. code:: bash

   #!/bin/bash
   svaba -t tumor.bam -n normal.bam -k 22 -G ref.fa -a test_id -p -4

使用如下指令提交：

.. code:: bash
      
   $ sbatch svaba.slurm

使用conda安装
----------------

.. code:: bash

   module load miniconda3
   conda create -n mypy
   source activate mypy
   conda install -c bioconda svaba
