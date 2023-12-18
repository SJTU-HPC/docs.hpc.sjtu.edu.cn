.. _Ncbi-rmblastn:

NCBI-RMBLASTN
=============================

简介
----
BLAST是“局部相似性基本查询工具”（Basic Local Alignment Search Tool）的缩写。
是由美国国立生物技术信息中心（NCBI）开发的一个基于序列相似性的数据库搜索程序。
该程序将DNA/蛋白质序列与公开数据库所有序列进行匹配比对，从而找到相似序列。

安装命令
------------

以下给出在Pi2.0平台通过spack安装该软件的命令：

.. code:: bash

   spack install ncbi-rmblastn@2.11.0 %gcc@8.5.0  

以下给出在Pi2.0平台编译安装该软件的命令：

.. code:: bash

   tar zxvf rmblast-1.0-ncbi-blast-2.2.28+.tar.gz
   cd rmblast-1.0-ncbi-blast-2.2.28+-src/c++/
   srun -n 4 -p cpu --pty /bin/bash
   ./configure --with-mt --prefix=path/to/rmblastn --without-debug
   make
   make install

参考资料
--------

-  `RMBlast <https://www.repeatmasker.org/rmblast/>`__
-  `Rmblastn Readme <https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/scripts/projects/rmblastn/README>`__
-  `Rmblastn tar <https://ftp.ncbi.nlm.nih.gov/blast/executables/rmblast/>`__