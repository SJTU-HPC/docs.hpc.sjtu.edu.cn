.. _Gatk:

GATK
======

简介
----
GATK是GenomeAnalysisToolkit的简称，是一系列用于分析高通量测序后基因突变的工具集合。它提供一种工作流程，
称作“ GATK Best Practices”。

CPU 容器版GATK
---------------

使用CPU容器版GTAK时，需要先指定GATK镜像的路径。然后使用 ``singularity exec 镜像路径 GTAK命令`` 的方式调用容器版GATK。

示例脚本如下(gatk-container.slurm)：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=test       
   #SBATCH --partition=cpu      
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   export IMAGE_NAME=/lustre/share/img/gatk-4.2.2.0.sif
   singularity exec $IMAGE_NAME gatk --java-options "-Xmx128G" ...


使用如下指令提交：

.. code:: bash
   
   $ sbatch gatk-container.slurm


.. _ARM版本GATK:


ARM Spack版GATK
-----------------

示例脚本如下(gatk.slurm):    

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

   $ sbatch gatk.slurm


ARM 容器版GATK
---------------

使用容器版GTAK时，需要先指定GATK镜像的路径。然后使用 ``singularity exec 镜像路径 GTAK命令`` 的方式调用容器版GATK。

示例脚本如下(gatk-container.slurm)：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g       
   #SBATCH -N 1          
   #SBATCH --ntasks-per-node=128
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   export IMAGE_NAME=/lustre/share/singularity/aarch64/gatk/gatk-4.2.0.0.sif
   singularity exec $IMAGE_NAME gatk --java-options "-Xmx128G" ...


使用如下指令提交：

.. code:: bash
   
   $ sbatch gatk-container.slurm

