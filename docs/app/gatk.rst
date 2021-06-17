.. _Gatk:

GATK
======

简介
----
GATK是GenomeAnalysisToolkit的简称，是一系列用于分析高通量测序后基因突变的工具集合。它提供一种工作流程，
称作“ GATK Best Practices”。

.. _ARM版本GATK:


ARM版GATK
------------

示例脚本如下(gromacs.slurm):    

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module use /lustre/share/spack/modules/kunpeng920/linux-centos7-aarch64
   module load gatk/4.2.0.0-gcc-9.3.0-openblas

   gatk --java-options "-Xmx128G" ...

使用如下指令提交：

.. code:: bash

   $ sbatch gromacs.slurm
