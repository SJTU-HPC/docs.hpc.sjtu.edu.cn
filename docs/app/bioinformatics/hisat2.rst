.. _Hisat2:

HISAT2
======

简介
----
HISAT2 是一种快速、灵敏的比对程序，用于将下一代测序读数（全基因组、转录组和外显子组测序数据）
与普通人群（以及单个参考基因组）进行映射。基于GCSA（图的BWT的扩展），我们设计并实现了图 FM 
索引（GFM），这是一种原始方法，也是我们所知的第一个实现。除了使用一个代表一般人群的全局 GFM 索引
外，HISAT2 还使用了一组大型的小型 GFM 索引，它们共同覆盖了整个基因组（每个索引代表一个 56 Kbp 
的基因组区域，需要 55,000 个索引来覆盖人群）。这些小索引（称为本地索引）与多种对齐策略相结合，
可以实现测序读数的有效对齐。

CPU 版本 Hisat2
---------------------------

.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=test
    #SBATCH --partition=cpu
    #SBATCH -N 1 
    #SBATCH --ntasks-per-node=40
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    
    module load hisat2/2.1.0-intel-19.0.4
    hisat2-build –p 4 genome.fa genome

使用如下指令提交：

.. code:: bash

    sbatch Hisat2.slurm

ARM 版本 Hisat2
---------------------------

.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=test
    #SBATCH --partition=arm128c256g
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=128
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    
    module use /lustre/share/spack/modules/kunpeng920/linux-centos7-aarch64
    module load hisat2/2.1.0-gcc-9.3.0
    hisat2-build –p 4 genome.fa genome


使用如下指令提交：

.. code:: bash

    sbatch Hisat2.slurm