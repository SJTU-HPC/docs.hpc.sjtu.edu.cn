*****
系统
*****

π 集群有文件系统和计算系统。


文件系统
========

π 集群采用 Lustre 并行文件系统作为后端存储系统。生产环境中已上线多套 lustre，挂载在计算节点的不同目录下：/lustre、/scratch。


全闪存并行文件系统
------------------

/scratch 目录挂载了全闪存并行文件系统，共 108T 容量。

系统特性
~~~~~~~~

该系统使用全套的 SSD（NVMe协议） 硬盘搭建，性能较高。单客户端最大读带宽达 5.7GB/s、最大写带宽达 10GB/s，4k 小文件的 IOPS 达到了读 170k、写 126k。

但是，由于 SSD 盘成本较高，因此提供的容量较小；同时在系统搭建时未设置高可用和数据备份，也存在数据存储安全性不高等问题。

基于该套系统的特性，推荐将其作为临时工作目录，可用于

1. 存储计算过程中产生的临时文件

2. 保存读写频率高的文件副本


**注意：为了保持系统的稳定可用，/scratch 目录每 3 个月会进行一次清理。因此，请务必及时将重要数据保存回 /lustre 个人目录。**

如何使用
~~~~~~~~

用户可以在以下路径找到 /scratch 提供的暂存空间： 
``/scratch/home/acct-xxxx/yyyy``

其中acct-xxxx代表计费账号（课题组目录），yyyy代表个人目录。

也可通过以下命令直接查看：

.. code:: bash

  $ echo $HOME | sed 's/lustre/scratch/'


为了快捷访问该目录，推荐设置软链接或设置环境变量：

1. 设置软链接

.. code:: bash
   
   $ ln -s $(echo $HOME | sed 's/lustre/scratch/') $HOME/scratch

然后，``cd $HOME/scratch`` 即可进入该临时工作目录。

2. 设置环境变量

.. code:: bash

   $ export SCRATCH=$(echo $HOME | sed 's/lustre/scratch/')

然后，``cd $SCRATCH`` 即可进入该临时工作目录。

注意，该方式只能临时设置 SCRATCH 变量，登录新的 shell 窗口时，该变量设置不生效。

若要长久保存变量，可编辑 $HOME/.bashrc 文件，添加：

.. code:: bash

   export SCRATCH=$(echo $HOME | sed 's/lustre/scratch/')

然后，进入新 shell 窗口变量生效 或者 ``source $HOME/.bashrc`` 在原 shell 窗口启用变量。