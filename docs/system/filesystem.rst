********
文件系统
********

HPC+AI 平台集群（除思源一号外）采用 Lustre 作为后端存储系统。Lustre是一种分布式的、可扩展的、高性能的并行文件系统，能够支持数万客户端、PB级存储容量、数百GB的聚合I/O吞吐，非常适合众多客户端并发进行大文件读写的场合。
Lustre最常用于高性能计算HPC，世界超级计算机TOP 10中的70%、TOP 30中的50%、TOP 100中的40%均部署了Lustre。

HPC+AI 平台集群（除思源一号外）已上线多套 Lustre 文件系统，挂载在计算节点的不同目录：/lustre、/scratch。

数据传输节点（data.hpc.sjtu.edu.cn）还多挂载了一个 /archive。

思源一号为独立集群，使用Gpfs文件系统，共10P。系统包含4 台 DSS-G Server 节点，每台配置 2 块 300G HDD， 用于安装操作系统，安装配置 GPFS 集群及创建文件系统。文件系统 metadata 采用 3 副本冗余，文件系统 data 采用 8+2p 冗余。 

主文件系统
==========

/lustre 目录挂载的为 HPC+AI平台集群（除思源一号外）中的主文件系统，共 13.1P，用户的个人目录即位于该目录。

主文件系统特性
--------------

主文件系统主要使用 HDD 盘搭建，旨在提供大容量、高可用、较高性能的存储供用户使用。搭建过程中，使用 RAID 保障硬盘级别的数据安全，使用 HA（High Availability） 保障服务器级别的高可用。

用户的主要工作、重要数据都应该发生和存储在主文件系统。

如何使用主文件系统
------------------

用户通过个人账户登录计算节点（包括登录节点）之后，默认进入主文件系统，即 HOME 目录。可以在以下路径找到 /lustre 提供给用户的空间：

``/lustre/home/acct-xxxx/yyyy``

其中acct-xxxx代表计费帐号（课题组帐号），yyyy代表个人帐号。

通过 ``cd`` ``cd $HOME`` ``cd ~`` 等方式都可进入主目录。


全闪存文件系统
==============

/scratch 目录挂载的为 HPC+AI平台集群（除思源一号外）的全闪存并行文件系统，共 108T 容量，可用作用户的临时工作目录。

全闪存文件系统特性
------------------

全闪存文件系统使用全套的 SSD（NVMe协议） 硬盘搭建，旨在提供高性能的存储供用户使用，可更好地支持 IO 密集型作业。对系统来说，单客户端最大读带宽达 5.7GB/s，最大写带宽达 10GB/s；4k 小文件读 IOPS 达 170k，写 IOPS 达 126k。但同时，由于成本问题，系统提供的容量较小；在搭建时也未设置高可用和数据备份，存在数据存储安全性不高等问题。

基于该套系统的特性，推荐用户将其作为临时工作目录，可用于

1. 存储计算过程中产生的临时文件

2. 保存读写频率高的文件副本


**注意：为了保持全闪存文件系统的稳定可用，/scratch 目录每 3 个月会进行一次清理。因此，请务必及时将重要数据保存回 /lustre 目录。**

如何使用全闪存文件系统
----------------------

用户可以在以下路径找到 /scratch 提供的暂存空间：
``/scratch/home/acct-xxxx/yyyy``

其中acct-xxxx代表计费帐号（课题组帐号），yyyy代表个人帐号。

为了快捷访问，我们已经为用户设置好环境变量，``cd $SCRATCH`` 即可进入该临时工作目录。

作业使用示例
------------

我们使用生信 WES 分析流程为例，该流程从测序文件开始，经 bwa 比对、samtools处理，然后用 GATK 检测变异（部分步骤）。 原代码如下：

.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=WES
    #SBATCH --partition=cpu
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=40
    #SBATCH --exclusive

    echo "##### 加载相关软件 #####"
    module load bwa samtools
    module load miniconda3 && source activate 10_24  # gatk-4.1.9.0

    echo "##### 设置变量 #####"
    REFDIR=$HOME/med/annotation/gatk/hg19  # 参考基因组和注释文件目录
    SAMPLEDIR=$HOME/med/testnor  # 样本目录

    WORKDIR=$HOME/WES_TEST  # 工作目录
    TMPDIR=$WORKDIR/tmpdir  # 临时缓存目录
    mkdir -p $WORKDIR
    mkdir -p $TMPDIR

    SampleID=test  # 样本名

    cd $WORKDIR

    echo "##### bwa 比对 #####"
    bwa mem -M -t 40 \
    ${REFDIR}/ucsc.hg19.fasta \
    ${SAMPLEDIR}/${SampleID}_1.fastq.gz \
    ${SAMPLEDIR}/${SampleID}_2.fastq.gz \
    | gzip -3 > ${SampleID}_mem.sam

    echo "##### samtools 生成bam #####"
    samtools view -@ 40 -bS ${SampleID}_mem.sam \
    | samtools sort -@ 40 > ${SampleID}_mem.sorted.bam

    samtools index ${SampleID}_mem.sorted.bam

    echo "##### gatk 检测变异 #####"
    gatk ReorderSam \
    -I ${SampleID}_mem.sorted.bam \
    -O ${SampleID}_mem.sorted.reorder.bam \
    -R ${REFDIR}/ucsc.hg19.fasta \
    --TMP_DIR ${TMPDIR} \
    --VALIDATION_STRINGENCY LENIENT \
    --SEQUENCE_DICTIONARY ${REFDIR}/ucsc.hg19.dict \
    --CREATE_INDEX true

    gatk MarkDuplicates \
    -I ${SampleID}_mem.sorted.reorder.bam \
    -O ${SampleID}_mem.sorted.reorder.rmdup.bam \
    --TMP_DIR ${TMPDIR} \
    --REMOVE_DUPLICATES false \
    --ASSUME_SORTED true \
    --METRICS_FILE ${SampleID}_mem.sorted.reorder.markduplicates_metrics.txt \
    --OPTICAL_DUPLICATE_PIXEL_DISTANCE 2500 \
    --VALIDATION_STRINGENCY LENIENT \
    --CREATE_INDEX true

过程中，会产生许多中间文件和临时文件。因此，可利用 $SCRATCH 作为临时目录，加快分析过程。只需要把脚本中的 ``WORKDIR=$HOME/WES_TEST`` 修改为 ``WORKDIR=$SCRATCH/WES_TEST`` 即可。


归档文件系统
============

在 `data （data.hpc.sjtu.edu.cn）` 节点的目录 `/archive` 下挂载了挂挡存储，共 3P 容量，用来存储用户的不常用数据。

归档文件系统特性
----------------

归档文件系统主要使用机械硬盘搭建，可提供大容量、高可用的存储供用户使用。搭建过程中，使用 RAID 保障硬盘级别的数据安全，使用 HA（High Availability） 保障服务器级别的高可用。归档文件系统作为主文件系统的一个补充，主要提供给用户存储不常用的数据（冷数据），从而释放主文件系统的存储空间、缓解主文件系统的存储压力。

** 注意：和主文件系统以及全闪存文件系统不同，归档文件系统只能在 `data` 节点访问，无法在计算节点和登录节点访问，也就是说保存在该文件系统的数据不能在计算节点读取并参与计算，因此只推荐保存不常使用的数据。**


如何使用归档文件系统
--------------------

(1) 登录 `data` 节点

.. code::

    # ssh $USER@data.hpc.sjtu.edu.cn

(2) 进入归档文件系统

用户可以在以下路径找到 /archive 提供的个人存储空间: ``/archive/home/acct-xxxx/yyyy``

其中 acct-xxxx 代表计费帐号（课题组帐号），yyyy 代表个人帐号。

为了快捷访问，我们已经为用户设置好环境变量，``cd $ARCHIVE`` 即可进入。


(3) 将不常用文件移动到 `$ARCHIVE`

.. code::

    # rsync -avh -P --append-verify $DATA $ARCHIVE

推荐使用 rsync 移动数据，详细参数含义可使用 ``man rsync`` 命令查看。
