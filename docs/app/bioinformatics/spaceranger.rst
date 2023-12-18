.. _spaceranger:

Space Ranger
=============================

简介
----
Visium空间基因表达是一种先进的分子分析方法，可以通过总mRNA来分类组织。
Space Ranger则是一组分析工具，用于处理Visium空间基因表达数据，这些数据是通过明场和荧光显微镜图像得到的。
使用Space Ranger，用户能够在甲醛固定石蜡包埋（FFPE）和新鲜冰冻组织中绘制整个转录组，从而深入了解正常发育、疾病病理学以及临床翻译研究的新颖观点。

可用版本
-------------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 2.1.1  | |cpu|   |precompile| spaceranger/2.1.1 Pi2.0                                   |
+--------+---------+----------+-----------------------------------------------------------+

安装命令
------------

前往Space Ranger下载页面填写10x Genomics用户软件许可协议信息，获得下载链接

.. code:: bash

   tar -zxvf spaceranger-2.0.0.tar.gz
   export PATH=/PATH/TO/spaceranger-2.1.1:$PATH 

spaceranger mkfastq 命令需要使用软件bcl2fastq，版本需要在2.20以上，可以通过conda在本地自行安装与调用：

.. code:: bash

   module load miniconda3
   conda create -n bcl2fastq python=3.8
   conda install -c freenome bcl2fastq=2.20

任务脚本
--------------

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=spacerange
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load spaceranger/2.1.1

   ulimit -s unlimited
   ulimit -l unlimited
   
   spaceranger count --sample mysample --fastqs data.fastq --disable-ui


参考资料
--------

-  `Space Ranger <https://www.10xgenomics.com/support/software/space-ranger/downloads>`__
-  `Space Ranger Tutorials <https://www.10xgenomics.com/cn/support/software/space-ranger/tutorials>`__