.. _Bwa:

BWA
======

简介
----
BWA，即Burrows-Wheeler-Alignment Tool。BWA 是一种能够将差异度较小的序列比对到一个较大的参考基因组上的软件包。它由三个不同的算法：

BWA-backtrack: 是用来比对 Illumina 的序列的，reads 长度最长能到 100bp。-
BWA-SW: 用于比对 long-read ，支持的长度为 70bp-1Mbp；同时支持剪接性比对。
BWA-MEM: 推荐使用的算法，支持较长的read长度，同时支持剪接性比对（split alignments)，但是BWA-MEM是更新的算法，也更快，更准确，且 BWA-MEM 对于 70bp-100bp 的 Illumina 数据来说，效果也更好些。
对于上述三种算法，首先需要使用索引命令构建参考基因组的索引，用于后面的比对。所以，使用BWA整个比对过程主要分为两步，第一步建索引，第二步使用BWA MEM进行比对。

bwa的使用需要两中输入文件：

Reference genome data（fasta格式 .fa, .fasta, .fna）
Short reads data (fastaq格式 .fastaq, .fq)



参考链接：https://www.jianshu.com/p/19f58a07e6f4