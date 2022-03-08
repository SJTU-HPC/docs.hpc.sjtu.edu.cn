.. _label_transfer:

***********
数据传输
***********

π 集群的登录节点通过公网 IP 传输数据，理论速度上限约为 110 MB/s，但是考虑到登录节点为大家共享使用，因此实际传输速度会偏低。对于数据传输，我们为您提供如下解决方案：

1. 少量数据传输，超算平台提供了专门用于数据传输的节点 (data.hpc.sjtu.edu.cn)，可以直接使用 putty, filezilla 等客户端，或在本地使用 scp, rsync 命令向该节点发起传输请求（因安全策略升级，在 π 集群的终端上不支持 scp/rsync 的远程传输功能，所以需要从用户本地终端使用 scp/rsync 命令）。

2. 1TB-1PB数据传输，强烈建议您联系我们，将硬盘等存储设备送至网络信息中心进行传输。

3. 超过1PB的数据，请您与我们联系，由计算专员根据具体情况为您解决数据传输问题。

重要说明
=========

传输节点选择
-------------

1. **少量数据传输**：login 节点和 data 节点均可，但推荐使用 data 节点。

2. **大量数据传输**：强烈推荐 data 节点。原因：1. 不占用 login 节点资源；2. 多进程或多用户同时传输不会受限于 CPU。

即：

**1. π 集群登录、作业提交，请使用 ssh login.hpc.sjtu.edu.cn**

**2. π 集群数据传输，推荐使用 scp data.hpc.sjtu.edu.cn local**

传输节点使用限制
------------------

**传输节点仅用于批量数据传输，请勿在此节点上运行与数据传输无关的应用，如编译程序、管理作业、校验数据等。如果发现此类行为，中心将视情况取消相关帐号使用传输节点的权利。**

.. _label_transfer_speed:

传输速度
=========

超算平台内部网络链路的带宽均不低于10Gbps，可以支持1GB/s的并行传输速度。但请注意包括rsync，scp，winscp等工具在内，大部分传输方式都是基于ssh通信的，而单个ssh连接支持的最大传输速度约100~150MB/s，在不使用额外手段多进程并发的情况下，以上工具均无法突破这一速度上限。

测速示例：通过万兆网络向传输节点上传数据，单个rsync指令传输速度约120MB/s。

.. image:: img/004.png
   :alt: single rsync test

测速示例：并发10个rsync同时向传输节点上传数据则可以达到1GB/s的总和速度，基本占满全部的可用带宽。

.. image:: img/005.png
   :alt: multiple rsync test

请注意以上测速是在排除了其他速度限制因素的理想网络环境下获得的结果，用户实际传输时可能遇到CPU资源，外部网络带宽，磁盘IO性能等瓶颈，也可能因多用户同时传输竞争网络带宽导致速度受限。传输节点预装有网络监控工具bmon，用户可以利用该工具查看节点当前的网络使用情况，判断是否有其他用户在进行大批量的数据传输。

传输方式
=========

Windows 用户
-------------

Windows 用户可以使用 WinSCP 在 π 群集和您自己的计算机之间传输文件。如下图所示，填写节点的地址，SSH 端口，SSH 用户名，SSH 密码，然后点击 Login 进行连接。 使用 WinSCP 的方法类似于使用 FTP 客户端 GUI，如下图所示：

.. image:: img/winscp01.png
   :alt: winscp example
   :height: 423px
   :width: 626px
   :scale: 75%

Linux/Unix/Mac用户
--------------------

如果传输的对象为少量大文件，且目标环境上没有数据的历史版本，所有需要传输的文件都是首次传输，可以使用scp直接拷贝文件。

.. code:: bash

   # 假设超算用户expuser01在平台上个人目录为/lustre/home/acct-exp/expuser01
   # 外部主机地址为100.101.0.1，该用户在外部主机上拥有帐号local_user且个人目录为/home/local_user/

   # 示例1：该用户要将个人目录中的~/math.dat文件下载到外部主机上
   $ scp expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/math.dat local_user@100.101.0.1:/home/local_user/

   # 示例2：该用户将本地目录~/data的全部数据上传至超算平台个人目录下
   $ scp -r local_user@100.101.0.1:/home/local_user/data expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/

如果需要传输的对象为包含大量文件的目录，或者目标环境上已经存在差异较小的历史版本，建议使用rsync拷贝数据，rsync会对比源地址和目标地址的内容差异，然后进行增量传输：

.. code:: bash

   # 示例3：该用户将超算平台上个人目录~/data的数据下载到外部主机，请注意rsync不支持双远端传输，必须在目标主机上操作
   $ rsync --archive --partial --progress expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/data/ ~/download/

   # 示例4：该用户将外部主机上的~/upload/exp04.dat文件上传到超算平台个人目录中
   $ rsync --archive --partial --progress ~/upload/exp04.dat expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/
   # 如果用户的外部环境CPU资源丰富而网络带宽相对较低，可以尝试--compress参数启用压缩传输
   $ rsync --compress --archive --partial --progress ~/upload/exp04.dat expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/

多进程并发传输
---------------

如 :ref:`label_transfer_speed` 段落所述，无论scp还是rsync，本质都是基于ssh连接的数据传输，都会受到ssh的传输效率限制。以上的单进程传输方式即使没有其他瓶颈制约，也只能达到100~150MB/s的传输速度。但是可以并发多个scp/rsync进程分别传输不同的内容来进一步提高网络带宽利用效率。

scp，rsync本身都不支持多进程传输，因此需要利用外部指令并发多个scp/rsync进程，外部封装的方法有很多，这里仅提供一种利用xargs自动分配传输文件的方法，熟悉脚本的用户也可以自制脚本来更灵活地将传输任务分配给各个传输进程。

.. code:: bash

   # 示例：并发5个rsync进程从超算平台个人目录~/data下载数据到外部主机~/download/路径下
   $ ssh expuser01@data.hpc.sjtu.edu.cn ls /lustre/home/acct-exp/expuser01/data/ > remote_list.txt
   $ cat remote_list.txt
     001.dat
     002.dat
     003.dat
     004.dat
     005.dat
   $ cat remote_list.txt | xargs --max-args=1 --max-procs=5 --replace=% rsync --archive --partial expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/data/% ~/download/

**注意：如果没有事先配置好免密码登录，rsync发起连接会要求用户输入密码，上述并发场合则会导致并发失败。** 请参考 :ref:`label_no_password_login` 预先配置好密钥。建议在并发操作之前先用rsync单独拷贝一个小文件进行测试，确认过程中没有手动交互的需求再进行正式的并发传输。

并发数量请控制在 **10个进程以内** ，因为目前超算网络最高支持1GB/s的传输速度，而单个ssh进程上限是100MB/s，10个并发进程就已经足够占用全部带宽。

思源一号传输方式
================

从本地传输数据到思源一号
-----------------------
.. code:: bash

   $ scp -r data.tar.gz username@sylogin1.hpc.sjtu.edu.cn:~

从闵行超算传送数据到思源一号
---------------------------

方式一：使用scp跨节点传输，可以并发多个scp进程来突破cpu单核心的性能瓶颈

.. code:: bash

   $ ssh user@202.120.58.247
   $ scp -r xucg.tar.gz user@111.186.43.1:~/target/directory/

或者使用rsync，适合目的地已经有大部分数据，仅做增量更新的情况

.. code:: bash

   $ ssh user@202.120.58.247
   $ rsync -avr --progress README.md user@111.186.43.1:/dssg/home/acct-hpc/user/

方式二：在202.120.58.247上直接复制/移动文件，此节点上预置环境变量$HOME指向闵行存储家目录，$SIYUANHOME指向思源存储家目录，登录时默认路径为闵行存储家目录。此方法虽然操作简便，但不支持多线程传输，请勿并发多个cp进程。传输速度基本等同于方式一中的单个scp进程。

.. code:: bash

   $ ssh user@202.120.58.247
   $ cp example.dat $SIYUANHOME/data/
