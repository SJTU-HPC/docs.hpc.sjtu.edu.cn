.. _cellranger:

Cell Ranger
=============================

简介
----
Cell Ranger是一组分析流程，用于从单细胞数据中执行样本解多重复用、条形码处理、单细胞3'和5'基因计数、
V(D)J转录序列组装和注释，以及特征条形码分析。

可用版本
-------------

+--------+---------+----------+-----------------------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                                    |
+========+=========+==========+===========================================================+
| 7.2.0  | |cpu|   |precompile| cellranger/7.2.0 Pi2.0                                    |
+--------+---------+----------+-----------------------------------------------------------+

安装命令
------------

前往Cell Ranger下载页面填写10x Genomics用户软件许可协议信息，获得下载链接

.. code:: bash

   tar -zxvf cellranger-7.2.0.tar.gz
   export PATH=/PATH/TO/cellranger-7.2.0:$PATH 

任务脚本
--------------

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=cellrange
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load cellranger/7.2.0
   cellranger testrun --id=check_install


参考资料
--------

-  `Cell Ranger <https://www.10xgenomics.com/support/software/cell-ranger/downloads>`__
-  `Cell Ranger Tutorials <https://www.10xgenomics.com/support/software/cell-ranger/tutorials>`__
