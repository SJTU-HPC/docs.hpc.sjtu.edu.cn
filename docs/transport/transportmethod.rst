***************
数据传输方法
***************

本地向思源一号传输
===================

**Windows 用户**


Windows 用户可以使用 WinSCP 在集群和您自己的计算机之间传输文件。可至 \ `WinSCP 官网 <https://winscp.net/eng/index.php>`__\下载。

如下图所示，填写思源一号节点的地址，SSH 端口，SSH 用户名，SSH 密码，然后点击 Login 进行连接。 使用 WinSCP 的方法类似于使用 FTP 客户端 GUI，如下图所示：

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

π 2.0/思源一号集群向冷存储传输
==============================

冷存储系统挂载在 ``data`` 节点和 ``sydata`` 节点的 ``/archive`` 下，可以将不常用的数据转移到冷存储。

数据传输
--------

``rsync`` 支持增量传输、断点续传、文件校验等功能，建议用 ``rsync`` 命令拷贝数据。

**使用 tmux 会话**

如果直接在命令行执行数据传输命令，网络不稳定可能导致传输中断，因此建议先开启 tmux 会话，再传输数据。相应的命令为：

.. code:: bash

   # 开启 tmux 会话，将其命名为 transport
   tmux new -s transport

   # 分离会话
   tmux detach

   # 查看所有的 tmux 会话
   tmux ls

   # 回到名为 transport 会话
   tmux attach -t transport

   # 彻底关闭 transport 会话
   tmux kill-session -t transport

**使用 rsync 传输数据**

假设用户 expuser01 需要将 Lustre 个人目录下的数据
``$HOME/data/`` 搬运到冷存储下的个人目录
``$ARCHIVE/data``\ ，需要执行的命令为：

.. code:: bash

   # -a 表示保存所有元数据，-r 表示包含子目录，--progress 表示显示进展，其余可用参数见 rsync 文档
   rsync -ar --progress $HOME/data/ $ARCHIVE/data

数据校验
--------

数据传输可能受网络波动影响，建议在数据传输完成之后，通过数据校验确认数据完整。对于思源一号集群，向冷存储的传输受网络波动影响可能性更大，强烈建议完成数据校验。

对于少量文件，可以用 md5sum 校验。对于多级目录结构，可以用 md5deep 工具。

**md5deep 校验（推荐）**

``md5deep`` 比 ``md5sum`` 命令更加丰富，可以递归地检查整个目录树，为子目录中的每个文件生成 md5 值。
文件的数量和大小会影响 md5 值生成的速度，如遇到这一步耗时较长，请耐心等待。

假设用户 expuser01 需要为 ``$HOME/data/`` 下的子目录的每个文件生成 md5 值，需要执行以下命令：

.. code:: bash

   cd $HOME/data/

   # 传输之前，对子目录的每个文件生成 md5 值
   md5deep -rl ./* > file.md5deep

   # 通过 rsync 传输数据
   # ...

   # 传输之后校验数据，和 md5 值不匹配的文件会被输出
   md5deep -rx file.md5deep $ARCHIVE/data

**md5sum 校验**

``md5sum``
可以生成文件校验码，来发现文件传输（网络传输、复制、本地不同设备间的传输）异常造成的文件内容不一致的情况。文件的数量和大小会影响 md5 值生成的速度，如遇到这一步耗时较长，请耐心等待。

.. code:: bash

   # 传输之前，对 txt 文件生成 md5 校验码
   ls *.txt | xargs -i -P 5 md5sum {} > file.md5

   # 通过 rsync 传输
   # ...

   # 传输之后，生成 md5 校验码
   # ...

   # 传输完成后，比较传输前后 md5 校验码
   diff file1.md5 file2.md5

清理存储空间
------------

在完成数据传输、数据校验之后，可以清理原文件占用的存储空间。

.. danger::
    下面的命令将直接删除对应路径下所有的内容，删除之后无法恢复数据，请确认路径正确后再执行！

.. code:: bash

   # 假设原文件存储位置在 /lustre/home/acct-exp/expuser01/data/
   rm -rf /lustre/home/acct-exp/expuser01/data/
