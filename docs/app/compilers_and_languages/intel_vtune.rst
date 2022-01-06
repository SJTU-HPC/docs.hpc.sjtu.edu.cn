.. _intel_vtune:

Intel-Vtune
========================

英特尔® VTune™ Profiler 为 HPC、云、物联网、媒体、存储等优化应用程序性能、系统性能和系统配置。

加载预安装的Vtune组件
---------------------

可以用以下命令加载集群中已安装的Intel组件:

+-----------------+--------------------------+--------------------------+
| 版本            | 加载方式                 | 组件说明                 |
+=================+==========================+==========================+        
| intel-mpi-2019. | module load              | Intel MPI库，由gcc编译   |
| 4.243/gcc-9.2.0 | intel-mpi/2019.4.243     |                          |
+-----------------+--------------------------+--------------------------+
| intel-mpi-2019. | module load              | Intel MPI库，由gcc编译   |
| 6.154/gcc-9.2.0 | intel-mpi/2019.6.154     |                          |
+-----------------+--------------------------+--------------------------+


使用 vtune
---------------------------

这里，我们演示如何使用系统中的vtune。

加载环境：

.. code:: bash

   $ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
   

vtune命令
-----------------------

1、vtune 可执行文件有两种形式
   vtune-xxxx
   amplxe-xxxx
   具体用那种形式，可以通过“ which vtune ”,查看vtune路径下的可执行文件。
2、收集程序热点
   amplxe-cl -collect hotspots -r test_hot ./test
   通过上面的命令可以收集到test应用的热点数据，保存到test_hot 目录下。
3、查看图形化界面
   vtune-gui ../r000hs.vtune 
   通过以上命令可以打开vtune的图形化界面。
   注意: 登录计算节点是，"ssh -X cas001",打开x11转发机制。以及在本地下载xmanager软件。

其他操作可以查看官方文档。

出现如下页面,说明vtune启动成功。

.. image:: ../../img/vtune-1.jpg

选择你的项目路径，进行性能分析，如图:

.. image:: ../../img/vtune-2.jpg

vtune可以分析的性能类型: 性能热点、内存消耗、线程、CPU占比、IO性能等，如图所示:

.. image:: ../../img/vtune-3.jpg

以 r003macc 程序为例,分析得出,程序执行耗时5.229s,内存、三级缓存使用情况、本地内存使用情况、以及NUMA远程访问情况,如图所示:

.. image:: ../../img/vtune-4.jpg

以 r003macc 程序为例,查看内存读写负载均衡、本地内存使用情况:

.. image:: ../../img/vtune-5.jpg

以 r003macc 程序为例,查看线程使用，以及CPU使用情况:

.. image:: ../../img/vtune-6.jpg


参考资料
--------

-  `intel-parallel-studio <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html/>`__
