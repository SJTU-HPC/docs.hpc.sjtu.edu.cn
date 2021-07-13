.. _Hmmer:

HMMER
======

简介
----
HMMER 和 BLAST 类似，主要用于序列比对。HMMER是由Sean Eddy编写的用于序列分析的免费且常用的软件包。它的
一般用法是鉴定同源蛋白质或核苷酸序列，并进行序列比对。

.. _ARM版本HMMER:


在ARM超算HMMER编译
------------------

该软件适用于ARM超算的正式版暂未发布，请关注软件官方网站等待正式版发布。
若任务急需可申请计算节点后输入以下命令编译开发版。
注意：开发版本可能存在问题，请尽量等待正式版本发布后再进行安装。    

.. code:: bash

    srun -p arm128c256g -n 4 --pty /bin/bash
    cd /YOUR/PATH/TO/HMMER
    git clone -b h3-arm https://github.com/EddyRivasLab/hmmer.git   
    cd hmmer
    git clone -b develop https://github.com/EddyRivasLab/easel.git
    cd easel
    autoconf
    ./configure --prefix=/YOUR/PATH/TO/HMMER/hmmer/easel
    make
    make check 
    make install
    cd ..   
    autoconf
    ./configure --prefix=/YOUR/PATH/TO/HMMER/hmmer
    make
    make check 
    make install

