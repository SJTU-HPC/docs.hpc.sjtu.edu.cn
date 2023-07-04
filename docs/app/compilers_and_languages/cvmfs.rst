:orphan:

CernVM-FS
=========

简介
----

CernVM-FS (Cern Virtual Machine File System) 是一种分布式文件系统，它由 CERN（欧洲核子研究中心）开发，主要服务于大型强子对撞机（Large Hadron
Collider, LHC）项目的计算需求。但也可以用于任何需要在大规模分布式环境中分发软件的应用场景，比如高性能计算（HPC）。

CernVM-FS 的设计理念是“全球只读”，即在全球范围内，所有客户端都看到的是同一个文件系统视图，且文件系统是只读的。所有对文件系统的修改都在服务器端完成，然后通过 CernVM-FS 分发到全球的客户端。这样可以确保所有客户端都在运行相同版本的软件，避免了版本不一致导致的问题。

CernVM-FS 的安装以及使用说明
----------------------------

CernVM-FS 已经在思源一号以及 Pi2.0 集群安装，可以直接使用。如果您需要使用的应用未添加到 CernVM-FS 的仓库，请通过邮件与我们联系。下面以 root 应用为例，演示 CernVM-FS 中应用的使用。

1. CernVM-FS 使用 autofs 自动挂载远端文件系统，如果本地 /cvmfs 目录下面为空，需要手动触发挂载动作

.. code:: bash

  ll /cvmfs/sft.cern.ch
  ll /cvmfs/sw.hsf.org

2. 选择合适的修改环境变量的脚本并执行

.. code:: bash

   # 思源一号的 OS 版本为 CentOS 8
   source /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos8-gcc10-opt/setup.sh
   # Pi2.0 的 OS 版本为 CentOS 7
   source /cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc8-opt/setup.sh

3. 检查 root 环境变量是否生效

.. code:: bash

   root --help

4. 检查 root 是否正常使用

.. code:: bash

   cd ~
   cp /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos8-gcc10-opt/tutorials/basic.C .
   cp /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos8-gcc10-opt/tutorials/tree/ntuple1.C .

5. 运行 root，调用之前从 tutorials 复制的测试文件

.. code:: bash

   root
   root [0] .x basic.C
   # output...
   root [1] .x ntuple1.C
   # output...

参考资料
--------

-  `CernVM-FS <https://cvmfs.readthedocs.io/en/stable/index.html>`__
-  `SPI/LCG
   Release <https://lcgdocs.web.cern.ch/lcgdocs/lcgreleases/introduction/>`__
-  `Root
   Tutorial <https://root.cern/doc/master/group__tutorial__v7.html>`__
