.. _SvABA:

SvABA
=================

简介
------------

SvABA是一种使用全基因组局部组装检测测序数据中结构变异的方法。

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
