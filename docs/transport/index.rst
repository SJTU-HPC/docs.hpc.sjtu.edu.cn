.. _label_transfer:

***********
数据传输
***********

目前交我算HPC+AI集群包含两个存储池，其中 **lustre** 存储用于 ``small, cpu, huge, 192c6t, dgx2, arm128c256g`` 以及相应的debug队列，应当在 ``login, kplogin, data`` 系列节点上查看、使用存放的数据； **gpfs** 存储用于 ``64c512g, a100`` 以及相应的debug队列，应当在 ``sylogin, sydata`` 节点上查看、使用存放的数据。使用时请注意自己home目录的实际路径。

推荐使用 data 节点进行数据传输，不占用 login 节点资源并且多进程或多用户同时传输不会受限于 CPU。**data节点仅用于批量数据传输，请勿在此节点上运行与数据传输无关的应用，如编译程序、管理作业、校验数据等。如果发现此类行为，中心将视情况取消相关帐号使用传输节点的权利。**

传输节点
=========

思源一号data节点地址为sydata.hpc.sjtu.edu.cn

π 2.0/AI/ARM 集群data节点地址为data.hpc.sjtu.edu.cn


本地向思源一号传输
===================

**Windows 用户**


Windows 用户可以使用 WinSCP 在集群和您自己的计算机之间传输文件。如下图所示，填写思源一号节点的地址，SSH 端口，SSH 用户名，SSH 密码，然后点击 Login 进行连接。 使用 WinSCP 的方法类似于使用 FTP 客户端 GUI，如下图所示：

.. image:: img/winscp01.png
   :alt: winscp example
   :height: 423px
   :width: 626px
   :scale: 75%

登录后即可看见左右两栏文件。左侧是本地文件，右侧是集群上的文件。点击需要传输的文件进行拖动即可传输。

**Linux/Unix/Mac用户**


Linux/Unix/Mac用户可通过在终端中使用scp或rsync等命令传输。

1.如果传输的对象为少量大文件，且目标环境上没有数据的历史版本，所有需要传输的文件都是首次传输，可以使用scp直接拷贝文件。

$ scp -r [源文件路径] [目标路径]

2.如果需要传输的对象为包含大量文件的目录，或者目标环境上已经存在差异较小的历史版本，建议使用rsync拷贝数据，rsync会对比源地址和目标地址的内容差异，然后进行增量传输。

$ rsync --archive --partial --progress [源文件路径] [目标路径]


**如果[源文件路径]或[目标路径]位于思源一号集群上，则路径需使用以下格式：**

[用户名]@sydata.hpc.sjtu.edu.cn:[思源一号上的路径]

**如果[源文件路径]或[目标路径]位于本地，则不需要加[用户名]@主机名，可直接写文件路径。**

.. code:: bash

   # 假设用户expuser01在思源一号平台上个人目录为/dssg/home/acct-exp/expuser01
   # 本地个人目录为/home/local_user/（个人目录可以用~代替）

   # 示例1：将本地目录~/data的全部数据上传至思源一号dssg目录下
   $ scp -r /home/local_user/data/ expuser01@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-exp/expuser01/

   # 示例2：将dssg目录中的~/math.dat文件下载到本地个人目录
   $ scp expuser01@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-exp/expuser01/math.dat /home/local_user/

   # 示例3：将dssg目录~/data的数据下载到本地~/download目录，请注意rsync不支持双远端传输，必须在目标主机（这里即为本地）上操作
   $ rsync --archive --partial --progress expuser01@data.hpc.sjtu.edu.cn:/dssg/home/acct-exp/expuser01/data/ /home/local_user/download/

本地向π 2.0/AI/ARM集群传输
==========================

**Windows 用户**


使用 WinSCP，方法和本地向思源一号传输类似，只需要将节点地址改成data.hpc.sjtu.edu.cn。


**Linux/Unix/Mac用户**


方法和本地向思源一号传输类似。

.. code:: bash

   # 假设用户expuser01在π 2.0集群上个人目录为/lustre/home/acct-exp/expuser01

   # 示例4：将本地目录~/data的全部数据上传至lustre目录下
   $ scp -r /home/local_user/data/ expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/

   # 示例5：将lustre目录~/data的数据下载到本地~/download目录，请注意rsync不支持双远端传输，必须在目标主机上操作
   $ rsync --archive --partial --progress expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/data/ /home/local_user/download/

思源一号与π 2.0/AI/ARM集群互传
================================

如果是在lustre和dssg直接跨存储池搬运数据，可以任选data或者sydata节点发起传输。例如通过登录π2.0集群数据传输节点data.hpc.sjtu.edu.cn，使用scp或rsync命令进行传输：

$ scp -r [源文件路径] [目标路径]

$ rsync -avr --progress [源文件路径] [目标路径]

此时因为已经登录到了π2.0集群，π集群上的文件路径不用加前缀，而思源一号上的文件路径需要加前缀[用户名]@sydata.hpc.sjtu.edu.cn。

.. code:: bash

   # 示例6: 该用户将lustre个人目录下的数据~/data搬运到dssg个人目录~/data下
   $ ssh expuser01@data.hpc.sjtu.edu.cn
   $ scp -r /lustre/home/acct-exp/expuser01/data/ expuser01@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-exp/expuser01/data/

传输方案
===========

对于数据传输，我们为您提供如下解决方案：

1. 少量数据传输，集群提供了专门用于数据传输的节点 (data.hpc.sjtu.edu.cn, sydata.hpc.sjtu.edu.cn)，可以直接使用 putty, filezilla 等客户端，或在本地使用 scp, rsync 命令向该节点发起传输请求（因安全策略升级，在集群的终端上不支持 scp/rsync 的远程传输功能，所以需要从用户本地终端使用 scp/rsync 命令）。

2. 1TB-1PB数据传输，强烈建议您联系我们，将硬盘等存储设备送至网络信息中心进行传输。

3. 超过1PB的数据，请您与我们联系，由计算专员根据具体情况为您解决数据传输问题。


提高数据传输速度的技巧
=======================

集群内部网络链路的带宽均不低于10Gbps，可以支持1GB/s的并行传输速度。但请注意包括rsync，scp，winscp等工具在内，大部分传输方式都是基于ssh通信的，而单个ssh连接支持的最大传输速度约100~150MB/s，但是可以并发多个scp/rsync进程分别传输不同的内容来进一步提高网络带宽利用效率。

scp，rsync本身都不支持多进程传输，因此需要利用外部指令并发多个scp/rsync进程，外部封装的方法有很多，这里仅提供一种利用xargs自动分配传输文件的方法，熟悉脚本的用户也可以自制脚本来更灵活地将传输任务分配给各个传输进程。

.. code:: bash

   # 示例：并发5个rsync进程从集群lustre目录~/data下载数据到外部主机~/download/路径下
   $ ssh expuser01@data.hpc.sjtu.edu.cn ls /lustre/home/acct-exp/expuser01/data/ > remote_list.txt
   $ cat remote_list.txt
     001.dat
     002.dat
     003.dat
     004.dat
     005.dat
   $ cat remote_list.txt | xargs --max-args=1 --max-procs=5 --replace=% rsync --archive --partial expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/data/% ~/download/

**注意：如果没有事先配置好免密码登录，rsync发起连接会要求用户输入密码，上述并发场合则会导致并发失败。** 请参考 :ref:`label_no_password_login` 预先配置好密钥。建议在并发操作之前先用rsync单独拷贝一个小文件进行测试，确认过程中没有手动交互的需求再进行正式的并发传输。

并发数量请控制在 **10个进程以内** ，因为目前集群网络最高支持1GB/s的传输速度，而单个ssh进程上限是100MB/s，10个并发进程就已经足够占用全部带宽。
