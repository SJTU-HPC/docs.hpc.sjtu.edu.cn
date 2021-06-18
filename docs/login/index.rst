****
登录
****

获取 π 集群账号后，可通过浏览器登陆 \ `可视化平台 HPC Studio <../studio/index.html>`__\ ，也可通过传统的 SSH 登陆。下面将介绍 SSH 登陆方法。

通过 SSH 登录 π 集群
==========================

本文将向大家介绍如何通过 SSH 远程登录到 π 集群上。在阅读本文档之前，您需要具备 Linux/Unix、终端、MS-DOS、SSH
远程登录的相关知识，或者您可以阅读参考资料理解这些概念。

本文主要内容：

-  使用 SSH 登录 π 集群的注意事项；
-  首次登录准备，如信息采集、客户端下载、SSH 登录、SSH 文件传输、无密码登录等；
-  故障排除和反馈。

按照文档的操作说明将有助于您完成工作，谢谢您的配合！

注意事项
--------

-   π 集群账号仅限于同一课题组的成员使用，请勿将账号借给他人使用。
-  请妥善保管好您的账号密码，不要告知他人。 π 集群管理员不会要求您提供密码。
-  恶意的 SSH 客户端软件会窃取您的密码，请在官网下载正版授权 SSH 客户端软件。
-  登录 π 集群后，请不要跳转到其他登录节点。任务完成后请关闭 SSH 会话。
-  若无法登录，请检查输入密码或确认 IP 地址是否正确。您可以参考故障排除和反馈，将诊断信息发送给 \ `HPC 邮箱 <mailto:hpc@sjtu.edu.cn>`__\ 。

准备
----

通过 SSH 登录 π 集群，需要在客户端输入登录节点 IP 地址（或主机名），SSH 端口，SSH 用户名和密码。账号开通后您会收到以下内容的邮件：

::

   SSH login node: login.hpc.sjtu.edu.cn
   Username: YOUR_USERNAME
   Password: YOUR_PASSWORD

登录节点 IP 地址（或主机名）为 login.hpc.sjtu.edu.cn

SSH 端口为 22

下载客户端
----------

Windows
^^^^^^^

Windows 推荐使用 Putty 免费客户端，下载后双击即可运行使用。可至 \ `Putty 官网 <https://www.putty.org>`__\ 
下载。


Linux/Unix/Mac
^^^^^^^^^^^^^^

Linux / Unix / Mac 操作系统拥有自己的 SSH 客户端，包括 ssh, scp, sftp 等。

通过 SSH 登陆 π 集群
----------------------------

下面介绍通过 SSH 登陆

Windows用户
^^^^^^^^^^^

启动客户端 Putty，填写登录节点地址 login.hpc.sjtu.edu.cn，端口号采用默认值 22，然后点 Open 按钮，如下图所示：

.. image:: ../img/putty1.png

在终端窗口中，输入您的 SSH 用户名和密码进行登录：

.. image:: ../img/putty2.png


*提示：输入密码时，不显示字符，请照常进行操作，然后按回车键登录。*

Linux/Unix/Mac用户使用SSH
^^^^^^^^^^^^^^^^^^^^^^^^^

Linux / Unix / Mac 用户可以使用终端中的命令行工具登录。下列语句指出了该节点的IP地址、用户名和SSH端口。

.. code:: bash

   $ ssh YOUR_USERNAME@TARGET_IP

通过 SSH 传输文件
-----------------

登录节点资源有限，不推荐在登录节点直接进行大批量的数据传输。超算平台提供了专门用于数据传输的节点，登录该节点后可以通过rsync，scp等方式将个人目录下的数据下载到本地，或者反向上传本地数据到个人目录。详情请参考具体请参考 :ref:`label_transfer` 。

.. _label_no_password_login:

无密码登录
----------

*提示：“无密码登录”仅适用于使用 SSH 命令行工具的 Linux/ UNIX / Mac 用户*

“无密码登录”使您无需输入用户名和密码即可登录，它还可以作为服务器的别名来简化使用。无密码登录需要建立从远程主机（群集的登录节点）到本地主机（您自己的计算机）的SSH信任关系。建立信任关系后，双方将通过 SSH 密钥对进行身份验证。

首先，您需要在本地主机上生成的 SSH 密钥对。为安全期间，π 集群要求使用密码短语 (passphrase) 来保护密钥对。使用密码短语来保护密钥对，每次双方身份验证时都需要输入密码。

.. code:: bash

   $ ssh-keygen -t rsa

接下来屏幕会显示：

.. code:: bash

   Generating public/private rsa key pair.
   Enter file in which to save the key (/XXX/XXX/.ssh/id_rsa):   # 存储地址，默认回车即可
   Enter passphrase (empty for no passphrase):                   # 请设置密码短语，并记住。输入的时候屏幕无显示
   Enter same passphrase again:                                  # 再输入一遍密码短语

.. tips: 为何要设置含有密码短语的密钥对： 输入ssh-keygen时，会请求您输入一个密码短语，您应该输入一些难以猜到的短语。

在无密码短语的情况下，您的私钥未经加密就存储在您的硬盘上，任何人拿到您的私钥都可以随意的访问对应的SSH服务器。

ssh-keygen 将在 ~/.ssh 中生成一个密钥对，包含两个文件：id_rsa(需保留的私钥)，和id_rsa.pub可作为您的身份发送的公钥）。然后，使用
ssh-copy-id 将本地主机的公钥 id_rsa.pub添加到远程主机的信任列表中。实际上，ssh-copy-id 所做的就是将id_rsa.pub的内容添加到远程主机的文件 ~/.ssh/authorized_keys 中。

.. code:: bash

   （在自己电脑上）$ ssh-copy-id YOUR_USERNAME@TARGET_IP

若手动自行在服务器上添加 authorized_keys 文件，需确保 authorized_keys
文件的权限为 600：

.. code:: bash

   （在 π 集群上）$ chmod 600 ~/.ssh/authorized_keys

.. image:: ../img/sshfile.png


我们还可以将连接参数写入 ~/.ssh/config 中，以使其简洁明了。
新建或编辑文件 ~/.ssh/config：

.. code:: bash

   $ EDIT ~/.ssh/config

还需分配以下内容：
主机分配远程主机的别名，主机名是远程主机的真实域名或IP地址，端口分配 SSH 端口，用户分配 SSH 用户名。

::

   Host hpc
   HostName TARGET_IP
   User YOUR_USERNAME

您需要确保此文件的权限正确：

.. code:: bash

   $ chmod 600 ~/.ssh/config

然后，您只需输入以下内容即可登录 π 群集：

.. code:: bash

    $ ssh hpc

*当 SSH 密钥对发生泄漏，请立即清理本地电脑 .ssh
文件夹里的密钥对，并重新在本地生成密钥对（生成时请设置密码短语）。另外请删除 π 集群上的 ~/.ssh/authorized_keys 文件。*

如何重新生成密钥对
----------------------------------------------------

.. code:: bash

   （在 π 集群上）$ rm -f ~/.ssh/authorized_keys           # 清除服务器上原有的 authorized_keys
   （在自己电脑上）$ rm  ~/.ssh/id*                           # 清除本地 .ssh 文件夹中的密钥对
   （在自己电脑上）$ ssh-keygen -t rsa                        # 在本地重新生成密钥对。第二个问题，设置密码短语 (passphrase)，并记住密码短语
   （在自己电脑上）$ ssh-keygen -R login.hpc.sjtu.edu.cn      # 清理本地 known_hosts 里关于 π 集群的条目     
   （在自己电脑上）$ ssh-copy-id YOUR_USERNAME@TARGET_IP      # 将本地新的公钥发给服务器，存在服务器的 authorized_keys 文件里

SSH 重置 known_hosts
--------------------

|avater| 若遇到上方图片中的问题，请重置 known_hosts，命令如下：

.. code:: bash

   （在自己电脑上）$ ssh-keygen -R login.hpc.sjtu.edu.cn

调试 SSH 登录问题
-----------------

有多种原因可能会阻止您登录到 π 集群。

1. 连续多次错输密码会被临时封禁 1 小时。集群登陆节点设置了 fail2ban 服务，多次输入密码错误后会被临时封禁 1 小时。

2. 若在登陆节点运行计算密集的作业，程序会被自动查杀，您的账号会被加入到黑名单，并在 30-120 分钟内无法登陆。

若需重置密码，请使用或抄送账号负责人邮箱发送邮件到  \ `HPC 邮箱 <mailto:hpc@sjtu.edu.cn>`__\ ，我们将会在 1 个工作日内响应您的申请。 

排查登陆问题，还可以使用 ping 命令检查您的电脑和 π 集群连接状态。

.. code:: bash

   $ ping login.hpc.sjtu.edu.cn


登陆常掉线的问题
----------------

如果 SSH 客户端长时间静默后，SSH 服务器端会自动断开相关会话。要解决这个，需要调整 SSH 的 keepalive 值，设置一个较长的静默时长阈值。

Mac/Linux用户
^^^^^^^^^^^^^

对于 Mac/Linux 用户，并且使用操作系统原生的终端 (terminal)，需要修改 \ ``$HOME/.ssh/config``\ 。具体的，在文件中添加如下内容：

.. code:: bash

   Host pi-sjtu-login:
       HostName login.hpc.sjtu.edu.cn
       ServerAliveInterval 240

其中 ServerAliveInterval 后的值即为阈值，单位为秒，用户可根据需要自行调整。

或者为了对所有的服务器设置长静默阈值：

.. code:: bash

   Host *
       ServerAliveInterval 240

之后保持 \ ``config``\ 文件为只可读：

.. code:: bash

   chmod 600 ~/.ssh/config

Windows SSH 客户端用户
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这里我们以 Putty 为例。市面有不同的 SSH 客户端，您可以根据自身情况自行搜索您使用的 SSH 客户端的设置方法。

在 Putty的 Session 的属性中，\ ``Connection`` ->
``Sending of null packets to keep session active`` ->
``Seconds between keepalives (0 to turn off)``\ 后的文本框中，输入对应的值，如 240。

ARM节点登录
===========

ARM平台简介
-----------

该平台基于ARM CPU构建，共100个计算节点，与π 2.0共享文件系统，数据无需迁移，但由于CPU架构不同，计算应用和软件库都需要重新编译。

单节点配备有128核（2.6 GHz）、256 GB内存（16通道DDR4-2933）、240 GB本地硬盘，节点间采用IB高速互联，挂载Lustre并行文件系统。

采用SLURM作业调度，提交方式与π 2.0一致，即在原有集群上新增一个队列，新队列名称：arm128c256g

ARM 节点登录方式。
-----------------
-  使用 \ ``srun``\ 登录命令：

.. code:: bash

   $ srun -p arm128c256g -n 4 --pty /bin/bash

-  或使用 \ ``salloc``\ 命令登录

.. code:: bash

   $ salloc -p arm128c256g -n 4
   $ ssh [分配的节点]

Tmux
====

Tmux是一个终端复用器（terminal multiplexer）。如果您有使用screen的经历的话，您可以理解为Tmux是screen的不同实现软件。本教程将讲解Tmux的基础用法。

Tmux是什么？
----------------

会话与进程
^^^^^^^^^^^^^^^^^^^

命令行的典型用法是打开终端（terminal）后，在里面输入指令。用户的这种与计算机交互的手段，称为\ **会话**\ （session）。

在会话中，通过命令行启动的所有进程均与会话进程绑定。当会话进程终止时，该会话启动的所有进程也会随之强行结束。

一点最常见的例子就是通过SSH连接到远程计算机。当SSH连接因为网络等原因断开时，那么SSH会话就被终止，这次会话启动的任务也会被强制结束。

为了解决这个问题，一种手段就是用户终端窗口与会话“解绑”。即关闭用户端窗口，仍然维持该会话，进而保证用户进程不变。

Tmux的作用
^^^^^^^^^^^^^^^^^^^

Tmux就是这样一款会话与窗口的“解绑”工具。

::

   （1）它允许在单个窗口中，同时访问多个会话。这对于同时运行多个命令行程序很有用。

   （2）它可以让新窗口"接入"已经存在的会话。

   （3）它允许每个会话有多个连接窗口，因此可以多人实时共享会话。

   （4）它还支持窗口任意的垂直和水平拆分

基本用法
------------

安装
^^^^^^^^^^^^^^^^^^^

π 集群中已经默认安装了Tmux，无须操作。如果您需要在自己的服务器上安装Tmux，请参考以下指令：

.. code:: bash

   # Ubuntu 或 Debian
   $ sudo apt-get install tmux

   # CentOS 或 Fedora
   $ sudo yum install tmux

   # Mac
   $ brew install tmux

启动与退出
~~~~~~~~~~~~~~

直接在终端中键入\ ``tmux``\ 指令，即可进入Tmux窗口。

.. code:: bash

   $ tmux

上面命令会启动 Tmux
窗口，底部有一个状态栏。状态栏的左侧是窗口信息（编号和名称），右侧是系统信息。

.. image:: /img/tmux_1.png

按下\ ``Ctrl+d``\ 或者显式输入\ ``exit``\ 命令，就可以退出 Tmux 窗口。

.. code:: bash

   $ exit

快捷键
^^^^^^^^^^^^^^^^^^^

Tmux有大量的快捷键。所有的快捷键都要使用\ ``Ctrl+b``\ 作为前缀唤醒。我们将会在后续章节中讲解快捷键的具体使用。

会话管理
------------

新建会话
^^^^^^^^^^^^^^^^^^^

第一个启动的会话名为\ ``0``\ ，之后是\ ``1``\ 、\ ``2``\ 一次类推。

但是有时候我们希望为会话起名以方便区分。

.. code:: bash

   $ tmux new -s SESSION_NAME

以上指令启动了一个名为\ ``SESSION_NAME``\ 的会话。

分离会话
^^^^^^^^^^^^^^^^^^^

如果我们想离开会话，但又不想关闭会话，有两种方式。按下\ ``Ctrl+b d``\ 或者\ ``tmux detach``\ 指令，将会分离会话与窗口

.. code:: bash

   $ tmux detach

后面一种方法要求当前会话无正在运行的进程，即保证终端可操作。我们更推荐使用前者。

查看会话
^^^^^^^^^^^^^^^^^^^

要查看当前已有会话，使用\ ``tmux ls``\ 指令。

.. code:: bash

   $ tmux ls

接入会话
^^^^^^^^^^^^^^^^^^^

``tmux attach``\ 命令用于重新接入某个已存在的会话。

.. code:: bash

   # 使用会话编号
   $ tmux attach -t 0

   # 使用会话名称
   $ tmux attach -t SESSION_NAME

杀死会话
^^^^^^^^^^^^^^^^^^^

``tmux kill-session``\ 命令用于杀死某个会话。

.. code:: bash

   # 使用会话编号
   $ tmux kill-session -t 0

   # 使用会话名称
   $ tmux kill-session -t SESSION_NAME

切换会话
^^^^^^^^^^^^^^^^^^^

``tmux switch``\ 命令用于切换会话。

.. code:: bash

   # 使用会话编号
   $ tmux switch -t 0

   # 使用会话名称
   $ tmux switch -t SESSION_NAME

``Ctrl+b s``\ 可以快捷地查看并切换会话

重命名会话
^^^^^^^^^^^^^^^^^^^

``tmux rename-session``\ 命令用于重命名会话。

.. code:: bash

   # 将0号会话重命名为SESSION_NAME
   $ tmux rename-session -t 0 SESSION_NAME

对应快捷键为\ ``Ctrl+b $``\ 。

窗格（window）操作
----------------------

Tmux可以将窗口分成多个窗格（window），每个窗格运行不同的命令。以下命令都是在Tmux窗口中执行。

划分窗格
^^^^^^^^^^^^^^^^^^^

``tmux split-window``\ 命令用来划分窗格。

.. code:: bash

   # 划分上下两个窗格
   $ tmux split-window

   # 划分左右两个窗格
   $ tmux split-window -h

.. image:: /img/tmux_2.png

对应快捷键为\ ``Ctrl+b "``\ 和\ ``Ctrl+b %``

移动光标
^^^^^^^^^^^^^^^^^^^

``tmux select-pane``\ 命令用来移动光标位置。

.. code:: bash

   # 光标切换到上方窗格
   $ tmux select-pane -U

   # 光标切换到下方窗格
   $ tmux select-pane -D

   # 光标切换到左边窗格
   $ tmux select-pane -L

   # 光标切换到右边窗格
   $ tmux select-pane -R

对应快捷键为\ ``Ctrl+b ↑``\ 、\ ``Ctrl+b ↓``\ 、\ ``Ctrl+b ←``\ 、\ ``Ctrl+b →``\ 。

窗格快捷键
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   $ Ctrl+b %：划分左右两个窗格。
   $ Ctrl+b "：划分上下两个窗格。
   $ Ctrl+b <arrow key>：光标切换到其他窗格。<arrow key>是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键↓。
   $ Ctrl+b ;：光标切换到上一个窗格。
   $ Ctrl+b o：光标切换到下一个窗格。
   $ Ctrl+b {：当前窗格左移。
   $ Ctrl+b }：当前窗格右移。
   $ Ctrl+b Ctrl+o：当前窗格上移。
   $ Ctrl+b Alt+o：当前窗格下移。
   $ Ctrl+b x：关闭当前窗格。
   $ Ctrl+b !：将当前窗格拆分为一个独立窗口。
   $ Ctrl+b z：当前窗格全屏显示，再使用一次会变回原来大小。
   $ Ctrl+b Ctrl+<arrow key>：按箭头方向调整窗格大小。
   $ Ctrl+b q：显示窗格编号。

.. |avater| image:: ../img/knownhosts.png


参考资料
========

-  http://www.ee.surrey.ac.uk/Teaching/Unix/
-  http://vbird.dic.ksu.edu.tw/linux_server/0310telnetssh.php#ssh_server
-  http://nerderati.com/2011/03/simplify-your-life-with-an-ssh-config-file/
-  http://www.cyberciti.biz/faq/ssh-passwordless-login-with-keychain-for-scripts/
-  https://stackoverflow.com/questions/25084288/keep-ssh-session-alive
-  https://patrickmn.com/aside/how-to-keep-alive-ssh-sessions/
-  https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/
-  https://danielmiessler.com/study/tmux/
-  https://linuxize.com/post/getting-started-with-tmux/
-  https://www.ruanyifeng.com/blog/2019/10/tmux.html
