.. _glibc:


glibc
===========

简介
----

glibc是GNU发布的libc库，即c运行库。 glibc是linux系统中最底层的api，几乎其它任何运行库都会依赖于glibc。 glibc除了封装linux操作系统所提供的系统服务外，它本身也提供了许多其它一些必要功能服务的实现。

glibc涉及的系统组件多，无法在超算平台上整体部署高版本glibc，如果需要高版本glibc，可通过源码自行安装。

安装代码
-----------

.. code-block:: bash

    srun -p 64c512g -n 16 --pty /bin/bash
    mkdir -p ${HOME}/01.application/12.glibc && cd ${HOME}/01.application/12.glibc
    wget http://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
    tar -zxvf glibc-2.29.tar.gz && cd glibc-2.29
    mkdir build && cd build
    ../configure --prefix=${HOME}/01.application/12.glibc/ --disable-sanity-checks
    make -j16
    make install
    cd ${HOME}/01.application/12.glibc/
    rm -rf glibc-2.29
    export PATH=${HOME}/01.application/12.glibc/bin:$PATH
    ldd --version  
   
