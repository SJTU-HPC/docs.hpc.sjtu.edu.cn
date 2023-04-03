CASM
======

简介
----

CASM是一款用于研究多成分晶体固体的第一原理统计力学软件

软件安装
----------

此项目为开源软件，建议使用conda进行安装


1.申请计算节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由于软件安装需要较多资源，登陆节点的资源限制会导致安装任务无法正常进行，因此需要先申请计算节点资源，在计算节点上进行安装。

2.conda安装
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
  
   conda create -n casm   --override-channels -c prisms-center -c conda-forge   casm-cpp=1.2.0 python=3
   
3.确认安装结果
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
  
   source activate casm
   ccasm --version
   #可以正常输出版本即为成功安装

