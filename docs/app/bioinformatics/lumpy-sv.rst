.. _Lumpy-sv:

LUMPY-SV
===========================

简介
----

lumpy是目前比较流行的一款SV检测工具，它同时支持PEM与SR和RD三种模式。在biostar上很多用户推荐，在lumpy所发的文章中，
与Pindel，delly，gasvpro等软件比较，也有不错的效果。软件使用也非常容易，不仅支持gemrline样品，也支持somatic样品。


ARM集群上的LUMPY-SV
-----------------------------

ARM上的LUMPY-SV以容器的形式安装部署。镜像路径为 ``/lustre/share/img/aarch64/lumpy-sv/lumpy-sv.sif`` 。

示例脚本如下(lumpy.slurm)：

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=test       
   #SBATCH --partition=arm128c256g      
   #SBATCH -N 1         
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   export IMAGE_NAME=/lustre/share/img/aarch64/lumpy-sv/lumpy-sv.sif
   singularity exec $IMAGE_NAME lumpyexpress \
    -B my.bam \
    -S my.splitters.bam \
    -D my.discordants.bam \
    -o output.vcf


使用如下指令提交：

.. code:: bash
   
   $ sbatch lumpy.slurm

