.. _easyfuse:

EasyFuse
=========

简介
----

EasyFuse是预测临床肿瘤样本特异性基因融合的新工具，包含了STAR-Fusion、InFusion、MapSplice2、Fusioncatcher 和 SoapFuse多种融合基因预测软件。

可用的版本
----------

+-----------+---------+----------+---------------------------------------------------------+
| 版本      | 平台    | 构建方式 | 路径                                                    |
+===========+=========+==========+=========================================================+
| 1.3.6     | |cpu|   | 容器     | /dssg/share/imgs/easyfuse/easyfuse_1.3.6.sif 思源一号   |
+-----------+---------+----------+---------------------------------------------------------+
| 1.3.5     | |cpu|   | 容器     | /dssg/share/imgs/easyfuse/easyfuse_1.3.5.sif 思源一号   |
+-----------+---------+----------+---------------------------------------------------------+
| 1.3.6     | |cpu|   | 容器     | /lustre/share/img/x86/easyfuse/easyfuse_1.3.6.sif       |
+-----------+---------+----------+---------------------------------------------------------+
| 1.3.5     | |cpu|   | 容器     | /lustre/share/img/x86/easyfuse/easyfuse_1.3.5.sif       |
+-----------+---------+----------+---------------------------------------------------------+

运行示例
--------

思源一号集群 EasyFuse
^^^^^^^^^^^^^^^^^^^^^^

输入的 ``/dssg/share/sample/easyfuse`` 为demo数据。

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=esayFuse_demo
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   singularity exec -e -B /dssg/share/data/easyfuse_ref:/ref \
    /dssg/share/imgs/easyfuse/easyfuse_1.3.5.sif \
    python /code/easyfuse/processing.py \
    -i /dssg/share/sample/easyfuse \
    -o output

使用如下脚本提交作业

.. code:: bash

   sbatch test.slurm


π 集群 EasyFuse
^^^^^^^^^^^^^^^^^

输入的 ``/lustre/share/samples/easyfuse`` 为demo数据。

EasyFuse@1.3.6
"""""""""""""""

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=esayFuse_demo
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   mkdir -p output

   singularity exec \
   --containall \
   -B /lustre/share/samples/easyfuse_ref:/ref \
   -B /lustre/share/samples/easyfuse:/data \
   -B $PWD/output:/output \
   /lustre/share/img/x86/easyfuse/easyfuse_1.3.6.sif \
   python /code/easyfuse/processing.py -i /data/ -o /output

EasyFuse@1.3.5
"""""""""""""""

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=esayFuse_demo
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   singularity exec -e -B /lustre/share/samples/easyfuse_ref:/ref \
    /lustre/share/img/x86/easyfuse/easyfuse_1.3.5.sif \
    python /code/easyfuse/processing.py \
    -i /lustre/share/samples/easyfuse \
    -o output

使用如下脚本提交作业

.. code:: bash

   sbatch test.slurm


输出结果
--------

.. code-block:: bash

   .
   ├── config.ini
   ├── easyfuse_processing.log
   ├── FusionSummary
   │   ├── SRR1659960_05pc_fusRank_1.csv
   │   ├── SRR1659960_05pc_fusRank_1.pred.all.csv
   │   └── SRR1659960_05pc_fusRank_1.pred.csv
   ├── process.sh
   ├── samples.db
   └── Sample_SRR1659960_05pc
       ├── expression
       ├── fetchdata
       ├── filtered_reads
       ├── fusion
       └── qc
   
参考资料
--------

-  `EasyFuse <https://github.com/TRON-Bioinformatics/EasyFuse>`__
