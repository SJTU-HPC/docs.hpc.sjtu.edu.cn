.. _Aspera:

Aspera
======

简介
----
Aspera通过使用获得专利的 Aspera 传输协议，Aspera 传输解决方案实现了距离无关性，能够在任何位置移动和存储数据，以及随时随地执行高速传输。

除了传输文件，Aspera 的流式技术还可以通过不受管理的互联网，在任何距离上接近零延迟地无错传输任何比特率的视频。

使用安全的用户访问控制、针对传输中的数据和静态存储的数据（可选）的 AES 数据加密以及数据完整性验证来防止中间人攻击，保护数据的安全。

Aspera安装步骤
---------------

调用conda

.. code:: bash

   $ module load miniconda3

创建conda环境，环境名可自定义，下文以“aspera”为例

.. code:: bash

   $ conda create -n aspera

激活conda环境

.. code:: bash

   $ conda activate aspera

安装apsera

.. code:: bash

   $ conda install -c hcc aspera-cli

Aspera下载示例
----------------

查看命令列表

.. code:: bash
   
  ascp -h
  主要使用参数：
  -v 详细模式
  -Q 用于自适应流量控制，磁盘限制所需
  -T 设置为无需加密传输
  -l 最大下载速度，一般设为500m
  -P TCP 端口，一般为33001
  -k 断点续传，通常设为 1
  -i 免密下载的密钥文件

下载

数据下载请在data节点上进行下载，激活环境后，可参考如下指令提交，

.. code:: bash

   $ ascp -vQT -l 50m -P33001 -k 1 -i ~/.conda/envs/ascp/etc/asperaweb_id_dsa.openssh era-fasp@fasp.sra.ebi.ac.uk:/vol1/srr/SRR182/094/SRR18231894 ./


era-fasp@fasp.sra.ebi.ac.uk:/vol1/srr/SRR182/094/SRR18231894 为示例下载站点及文件,下载路径为当前路径。

注意：如果是用户个人安装的conda需要根据env路径更改openssh文件路径，可通过"which ascp"或者"conda env list"查看aspera的conda环境
路径。
