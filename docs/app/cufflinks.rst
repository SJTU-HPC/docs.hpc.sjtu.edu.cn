.. _Cufflinks:

CUFFLINKS
=======================================

简介
----

Cufflinks下主要包含cufflinks,cuffmerge,cuffcompare和cuffdiff等几支主要的程序。
主要用于基因表达量的计算和差异表达基因的寻找。Cufflinks程序主要根据Tophat的比对结果，
依托或不依托于参考基因组的GTF注释文件，计算出(各个gene的)isoform的FPKM值，
并给出trascripts.gtf注释结果(组装出转录组)。