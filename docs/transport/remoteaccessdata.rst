.. _remoteaccessdata:

通过远程挂载读写超算平台的数据
===============================

要读写超算平台的数据，除了通过登录到集群的登录节点和计算节点直接操作数据，还可以通过
``sshfs`` 挂载家目录到 Windows 或 Linux
主机。下面介绍如何通过远程挂载的方式读写超算平台的数据。

``sshfs`` 介绍
--------------

``sshfs`` 是基于 fuse 构建的 ssh
文件系统客户端程序。无需对远程主机的配置作任何改变，就可以通过 ssh
协议来挂载远程文件系统。

挂载家目录到 Linux 主机
-----------------------

假设用户 expuser01 现在需要将 Pi 2.0 集群上自己的家目录
``/lustre/home/acct-exp/expuser01``\ 挂载到本地的 ``/mountpoint1``\ ，
以及将思源集群上自己的家目录 ``/dssg/home/acct-exp/expuser01``
挂载到本地的 ``/mountpoint2``，可以按照下面步骤操作：

.. code:: bash

   # 示例1：在 Linux 系统环境挂载 Pi 2.0 集群上的个人家目录（需要先安装 sshfs 软件）
   # sshfs -p 22 -o allow_other expuser01@data.hpc.sjtu.edu.cn:/lustre/home/acct-exp/expuser01/ /mountpoint1

   # 示例2：在 Linux 系统环境挂载思源一号集群上的个人家目录（需要先安装 sshfs 软件）
   # sshfs -p 22 -o allow_other expuser01@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-exp/expuser01/ /mountpoint2

挂载家目录到 Windows 主机
-------------------------

.. code:: bash

   # 示例3：在 Windows 系统环境挂载 Pi 2.0 集群上的个人家目录（需要先安装 WinFSP：https://github.com/billziss-gh/winfsp ，再安装sshfs-win：https://github.com/billziss-gh/sshfs-win ）
   # 在Windows的文件资源管理器中点击『映射网络驱动器』： 添加 \\sshfs\expuser01@202.120.58.253 后点击『连接』

   # 示例4：在 Windows 系统环境挂载思源一号集群上的个人家目录（需要先安装 WinFSP：https://github.com/billziss-gh/winfsp ，再安装sshfs-win：https://github.com/billziss-gh/sshfs-win ）
   # 在Windows的文件资源管理器中点击『映射网络驱动器』： 添加 \\sshfs\expuser01@111.186.43.4 后点击『连接』
