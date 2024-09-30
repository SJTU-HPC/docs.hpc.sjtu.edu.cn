****************
VS Code连接集群
****************

Visual Studio Code（简称VS Code）是一款由微软开发且跨平台的免费源代码编辑器。 该软件支持语法高亮、代码自动补全（又称IntelliSense）、代码重构功能，并且内置了命令行工具和Git 版本控制系统。

VS Code经过配置，可以远程连接到Pi集群及思源一号，在本地进行远程的开发部署工作。但 **请注意您依然需要有一个常规的SSH登录会话以向平台证明您处于活跃状态，否则您的vscode连接可能会被视为残留的僵尸进程而被自动清理掉** 。

安装兼容的SSH客户端
-----------------------
首先需要在本地电脑上安装OpenSSH兼容的SSH客户端（Putty不支持）。

对于Mac，系统自带的SSH客户端就可满足需求，无需安装。

对于linux用户，需要安装 `openssh-client`。

运行

.. code:: console

   $ sudo apt-get install openssh-client

或者

.. code:: console

   $ sudo yum install openssh-client


对于 Windows 用户，请安装 Windows OpenSSH Client。Windows 用户可以使用 Windows 设置或者 PowerShell 来安装该客户端，具体请参考链接 `安装 OpenSSH <https://docs.microsoft.com/zh-cn/windows-server/administration/openssh/openssh_install_firstuse>`_。


使用密码登录集群
----------------------

在vscode中选择 `连接到主机` 并发起SSH连接后，右下角会显示一个正在连接中的提示小窗，点击提示信息中高亮的 `details` ，vscode会给您切换到终端标签页，在这里您将被要求输入密码。如果一段时间没有完成密码输入，vscode会按超时处理放弃连接。

和SSH客户端进行登录一样，如果您已经绑定了jAccount，您可以在这里使用jAccount密码登录。


免密登录集群
-----------------------

要进行免密登录首先需要一份在有效期内的免密证书，申请免密证书请参考 :ref:`此章节<require_certificate>`。

获取到免密证书后，您需要在本地的ssh会话配置中指定使用它。本地配置文件的位置取决于您的系统环境，linux环境通常在 `~/.ssh/config` ，windows环境通常在 `C:\Users\[YOUR_USERNAME]\.ssh\config` ，如果您不确定实际位置，可以从vscode的 `连接到主机 - 配置SSH主机` 选项菜单中查看官方给你的候选路径。以下是一个配置实例，请根据自己的实际情况变通：

.. code:: bash

   Host bob_want_password_free_login
     HostName pilogin.hpc.sjtu.edu.cn
     User bob
     Port 22
     IdentityFile C:\Users\Bob\Documents\MobaXterm\home\.ssh\id_ed25519
     CertificateFile C:\Users\Bob\Documents\MobaXterm\home\.ssh\id_ed25519-cert.pub

   Host charley_want_to_use_password
     HostName sylogin.hpc.sjtu.edu.cn
     User charley
     Port 22

配置完毕后请在本地终端测试是否能访问集群。

.. code:: console

   $ ssh bob_want_password_free_login

本地安装 VS Code 及插件（可选）
-----------------------------------

本章节介绍的插件可以优化使用体验，并非必须。

请至 `VS code download <https://code.visualstudio.com/download>`_ 下载于本地操作系统对应的 VS Code安装包并根据步骤安装。

打开VS Code软件， 安装 Remote SSH插件。

.. image:: /img/remote-ssh_install.png

安装完毕后点开左方工具栏中remote-ssh插件的图标，该插件会自动读取 `~/.ssh/config` 中的主机名。

.. image:: /img/remote-ssh-servers.png

右键相应的主机名即可选择连接主机：

.. image:: /img/remote-ssh-click.png

此时会弹出窗口要求输入先前设置的passphrase:

.. image:: /img/passphrase.png

输入密码后即可链接至远程主机：

.. image:: /img/remote-ssh-connection.png

连接后可选择打开文件夹或者终端：

.. image:: /img/remote-ssh-file-terminal.png
