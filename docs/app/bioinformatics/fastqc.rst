.. _fastqc:

FastQC
=========

简介
----

FastQC是一个二代测序数据质量控制软件。

可用的版本
----------

+-------------+---------+----------+-----------------------------------------------+
| 版本        | 平台    | 构建方式 | 模块名                                        |
+=============+=========+==========+===============================================+
| 0.11.9      | |cpu|   | Spack    | `fastqc/0.11.9-gcc-11.2.0-openjdk`_ 思源一号  |
+-------------+---------+----------+-----------------------------------------------+
| 0.11.7      | |cpu|   | Spack    | `fastqc/0.11.7-gcc-9.2.0`_                    |
+-------------+---------+----------+-----------------------------------------------+
| 0.11.7      | |arm|   | Spack    | fastqc/0.11.7-intel-19.0.4                    |
+-------------+---------+----------+-----------------------------------------------+
| 0.11.7      | |cpu|   | Spack    | fastqc/0.11.7-gcc-8.3.0                       |
+-------------+---------+----------+-----------------------------------------------+

使用 Conda 安装 FastQC
--------------------------

推荐使用 ``Conda`` 在用户目录部署特定的 ``FastQC`` 软件，以思源一号为例：

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module purge
   module load miniconda3/4.10.3
   conda create -n biotools                 # 创建新的环境
   source activate biotools                 # 激活环境
   conda install -c bioconda fastqc         # 安装软件
   fastqc --help

示例文件
--------

.. code-block:: bash

   # 思源一号
   /dssg/share/sample/bwa/B17NC_R1.fastq.gz
   /dssg/share/sample/bwa/B17NC_R2.fastq.gz
   # π 集群
   /lustre/share/samples/bwa/B17NC_R1.fastq.gz
   /lustre/share/samples/bwa/B17NC_R2.fastq.gz

运行示例
--------

.. _fastqc/0.11.9-gcc-11.2.0-openjdk:

思源一号集群 FastQC
^^^^^^^^^^^^^^^^^^^^^^

作业脚本 ``test.slurm`` 内容如下:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=QC
   #SBATCH --partition=64c512g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=10

   ulimit -l unlimited
   ulimit -s unlimited
   
   module load fastqc/0.11.9-gcc-11.2.0-openjdk
   input_dir=/dssg/share/sample/bwa
   fastqc -f fastq -o ~/QC $input_dir/B17NC_R1.fastq.gz $input_dir/B17NC_R2.fastq.gz

使用 ``sbatch`` 提交作业

.. code:: bash

   sbatch test.slurm


.. _fastqc/0.11.7-gcc-9.2.0:

π 集群 FastQC
^^^^^^^^^^^^^^^^^

作业脚本 ``test.slurm`` 内容如下:    

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=QC
   #SBATCH --partition=small
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=10

   ulimit -l unlimited
   ulimit -s unlimited

   module load fastqc/0.11.7-gcc-9.2.0
   input_dir=/lustre/share/samples/bwa
   fastqc -f fastq -o ~/QC $input_dir/B17NC_R1.fastq.gz $input_dir/B17NC_R2.fastq.gz

运行结果
--------

会输出质控网页报告，可下载后查看。

.. code-block:: bash

   QC
   ├── B17NC_R1_fastqc.html
   ├── B17NC_R1_fastqc.zip
   ├── B17NC_R2_fastqc.html
   └── B17NC_R2_fastqc.zip

FASTQ 格式说明
--------------

FASTQ文件是一个文本文件，其中包含通过流动槽 ``flow cell`` 上质控参数的簇 ``cluster`` 的测序数据。
对于每个通过质控参数的簇，一个序列被写入相应样本的 ``R1 FASTQ`` 文件，而对于双端测序运行，另外一个序列也被写入该样本的 ``R2 FASTQ`` 文件。 FASTQ文件中的每个条目包含4行：

1. 序列标识符，其中包含有关测序运行和簇的信息；
#. 序列（碱基信号； A，C，T，G和N）；
#. 分隔符，只是一个加号（+）；
#. 读取碱基的质量值。 这些是Phred +33编码的，使用ASCII字符表示数字质量值。

FASTQ文件中单个记录条目的示例：

.. code-block:: bash

   @SIM:1:FCX:1:15:6329:1045 1:N:0:2
   TCGCACTCAACGCCCTGCATATGACAAGACAGAATC
   +
   <>;##=><9=AAAAAAAAAA9#:<#<;<<<????#=


参考资料
--------

-  `FASTQ格式说明 <https://help.basespace.illumina.com/files-used-by-basespace/fastq-files>`__
