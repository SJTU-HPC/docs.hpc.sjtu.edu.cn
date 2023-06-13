***************
数据传输技巧
***************

提高数据传输速度的技巧(从lustre目录到外部主机)
==============================================

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

提高数据传输速度的技巧(从lustre目录到archive目录)
=================================================

利用rsync命令并发多个scp进程可有效提高数据传输的速度

.. code:: bash

   # 示例：并发5个scp进程从 pi 2.0 集群lustre目录/lustre/home/acct-hpc/expuser01/img/传送文件数据至归档存储/archive/home/acct-hpc/expuser01/sif/目录下
   ssh expuser01@data.hpc.sjtu.edu.cn
   find /lustre/home/acct-hpc/expuser01/img -type f | xargs -P 5 -I {} scp {} hpc@data.hpc.sjtu.edu.cn:/archive/home/acct-hpc/expuser01/sif/

上述命令启动5个scp进程并发传送47GB的文件数据，添加 ``time`` 命令可统计传送时间，

比如 ``time scp -r /lustre/home/acct-hpc/expuser01/img/* hpchgc@data.hpc.sjtu.edu.cn:/archive/home/acct-hpc/expuser01/sif/`` 

使用1、5个进程从lustre目录传送47GB的数据至archive目录所用时间如下所示

+--------+--------+
| 进程数 | 时间   |
+========+========+
| 1      | 10m17s |
+--------+--------+
| 5      | 3m47s  |
+--------+--------+

通过sshfs挂载家目录
===================

.. code:: bash

   # 假设用户expuser01在π 2.0集群上个人目录为/lustre/home/acct-exp/expuser01 ；思源一号的家目录为/dssg/home/acct-exp/expuser01

   # 示例1：在linux系统环境挂载π 2.0集群上的个人家目录（需要先安装sshfs软件）
   # sshfs -p 22 -o allow_other expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/ /mountpoint

   # 示例2：在linux系统环境挂载思源一号集群上的个人家目录（需要先安装sshfs软件）
   # sshfs -p 22 -o allow_other expuser01@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-exp/expuser01/ /mountpoint

   # 示例3：在windows系统环境挂载π 2.0集群上的个人家目录（需要先安装WinFSP：https://github.com/billziss-gh/winfsp ，再安装sshfs-win：https://github.com/billziss-gh/sshfs-win ）
   # 在Windows的文件资源管理器中点击『映射网络驱动器』： 添加 \\sshfs\expuser01@202.120.58.253 后点击『连接』

   # 示例4：在windows系统环境挂载思源一号集群上的个人家目录（需要先安装WinFSP：https://github.com/billziss-gh/winfsp ，再安装sshfs-win：https://github.com/billziss-gh/sshfs-win ）
   # 在Windows的文件资源管理器中点击『映射网络驱动器』： 添加 \\sshfs\expuser01@111.186.43.4 后点击『连接』

