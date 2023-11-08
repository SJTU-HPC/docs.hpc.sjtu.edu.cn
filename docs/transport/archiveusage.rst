.. _archiveusage:

冷存储数据迁移指南
==============================

冷存储系统挂载在 ``data`` 节点和 ``sydata`` 节点的 ``/archive`` 下，可以将不常用的数据转移到冷存储。
在转移 Pi 超算 ``/lustre`` 下的数据时，需要选择 data 传输节点，在转移思源一号 ``/dssg`` 下的数据时，需要选择 sydata 传输节点。
如果要使用冷存储中的数据，用户需要先将数据手动从冷存储传输到 Lustre/GPFS。

数据传输
--------

``rsync`` 支持增量传输、断点续传、文件校验等功能，建议用 ``rsync`` 命令拷贝数据。

**使用 tmux 会话**

如果直接在命令行执行数据传输命令，网络不稳定可能导致传输中断，因此建议先开启 tmux 会话，再传输数据。相应的命令为：

.. code:: bash

   # 开启会话：在终端输入下面命令，会开启名为 transport 的 tmux 会话
   tmux new -s transport

   # 分离会话：在会话中输入下面命令，会从会话返回终端
   # 如果会话在执行程序中无法输入命令，先按 Ctrl+B，再按 D
   tmux detach

   # 查看所有的 tmux 会话
   tmux ls

   # 回到名为 transport 会话
   tmux attach -t transport

   # 彻底关闭 transport 会话
   tmux kill-session -t transport

**使用 rsync 从 Pi 超算或思源一号向冷存储传输数据**

假设用户 expuser01 需要将 Pi 超算个人目录下的数据 ``$HOME/data/`` 搬运到冷存储下的个人目录 ``$ARCHIVE/data/``\ ，需要执行的命令为：

.. code:: bash

   # -a 表示保存所有元数据，-v 打印更多信息，-r 表示包含子目录，-P 表示显示进展以及保存未传输完的文件，其余可用参数见 rsync 文档
   rsync -avrP $HOME/data/ $ARCHIVE/data/

   # 如果要搬运的目录中包含软链接文件，需要用 --copy-unsafe-links 参数，将不安全的软链接（原文件不在备份范围内）转换成实体文件保存
   rsync -avrP --copy-unsafe-links $HOME/data/ $ARCHIVE/data/

   # 如果要搬运的目录中包含硬链接文件，可以用 -H 参数，保留文件之间的硬链接信息，减少不必要的文件传输
   rsync -avrPH $HOME/data/ $ARCHIVE/data/

**使用 rsync 从冷存储向 Pi 超算或思源一号传输数据**

假设用户 expuser01 需要将冷存储下的个人目录 ``$ARCHIVE/data/`` 搬运到 Pi 超算个人目录下的数据 ``$HOME/data/``\ ，需要执行的命令为：

.. code:: bash

   # -a 表示保存所有元数据，-v 打印更多信息，-r 表示包含子目录，-P 表示显示进展以及保存未传输完的文件，其余可用参数见 rsync 文档
   rsync -avrP $ARCHIVE/data/ $HOME/data/

   # 如果要搬运的目录中包含软链接文件，需要用 --copy-unsafe-links 参数，将不安全的软链接（原文件不在备份范围内）转换成实体文件保存
   rsync -avrP --copy-unsafe-links $ARCHIVE/data/ $HOME/data/

   # 如果要搬运的目录中包含硬链接文件，可以用 -H 参数，保留文件之间的硬链接信息，减少不必要的文件传输
   rsync -avrPH $ARCHIVE/data/ $HOME/data/

在 ``rsync`` 传输完成之后，会出现类似下面的提示信息：

.. code:: bash

    ...
    sent 5,244,160,230 bytes  received 137 bytes  48,333,275.27 bytes/sec
    total size is 5,242,880,013  speedup is 1.00

如果因为某些原因，导致 ``rsync`` 未正常结束，可以再次用 ``rsync`` 命令传输数据，``rsync`` 的断点续传功能可以接着上次传输进度继续传输。

数据校验
--------

数据传输可能受网络波动影响，建议在数据传输完成之后，通过数据校验确认数据完整。对于思源一号集群，向冷存储的传输受网络波动影响可能性更大，强烈建议完成数据校验。

对于少量文件，可以用 md5sum 校验。对于多级目录结构，可以用 md5deep 工具。

**md5deep 校验（推荐）**

``md5deep`` 比 ``md5sum`` 命令更加丰富，可以递归地检查整个目录树，为子目录中的每个文件生成 md5 值。
文件的数量和大小会影响 md5 值生成的速度，如遇到这一步耗时较长，请耐心等待。

假设用户 expuser01 需要为目录 ``$HOME/data/`` 和 ``$ARCHIVE/data/`` 下的子目录的每个文件生成 md5 值，需要执行以下命令：

.. code:: bash

   # 传输之前，对子目录的每个文件生成 md5 值
   md5deep -rl $HOME/data/ > before.md5deep

   # 通过 rsync 传输数据
   # ...

   # 传输之后，对目录的每个文件生成 md5 值
   md5deep -rl $ARCHIVE/data/ > after.md5deep

   # md5deep 文件的格式为“md5值 文件路径”，比较传输前后的各文件 md5 值需要使用第一列
   sort before.md5deep | awk '{print $1}' > before
   sort after.md5deep | awk '{print $1}' > after
   diff before after

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

在完成数据传输、数据校验之后，可以清理原文件占用的存储空间。冷存储一般用于存储不常用的数据，因此建议每次使用前将数据从冷存储传输到 Pi 超算或思源一号，在使用后再清理掉位于 Pi 超算或思源一号的数据。

* 如果要清理位于 Pi 超算或思源一号的数据，可以参考以下命令：

.. danger::
    下面的命令将直接删除对应路径下所有的内容，删除之后无法恢复数据，请确认路径正确后再执行！

.. code:: bash

   # 假设原文件存储位置在 /lustre/home/acct-exp/expuser01/data/
   rm -rf /lustre/home/acct-exp/expuser01/data/

* 如果要清理位于冷存储的数据，可以参考以下命令：

.. danger::
    下面的命令将直接删除对应路径下所有的内容，删除之后无法恢复数据，请确认路径正确后再执行！

.. code:: bash

   # 假设原文件存储位置在 /archive/home/acct-exp/expuser01/data/
   rm -rf /archive/home/acct-exp/expuser01/data/