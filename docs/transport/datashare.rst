.. _datashare:

************************
交我算账号的数据共享
************************


交我算的主帐号和子帐号都为独立帐号，仅在计费关系上存在关联。若有数据共享需求，可由主帐号负责人发邮件到\ `hpc邮箱 <mailto:hpc@sjtu.edu.cn>`__\ 。

::

    邮件模板：

    我是超算账号expuser01（主账号名）使用人，已经阅读超算账号数据共享文档，现需要在思源一号/Pi2.0集群建立共享文件夹。
    使用方式：课题组内部共享/课题组之间共享
    其他账号：expuser02，expuser03 （可选，若为课题组内部共享则无需填写）

.. note::
    π2.0/AI/ARM集群及思源一号集群使用两个独立的存储系统，因此共享文件夹需要分开申请。

数据共享场景
======================

目前数据共享场景包括：课题组内部共享（主账号和子账号的共享）、课题组之间共享（不同主账号之间的共享）、共享数据到集群外。

课题组内部共享
----------------------

- 如何申请：由主账号负责人向hpc邮箱发送邮件，申请在主账号下建立 ``acct-expuser01/share`` 文件夹。
- 权限控制：主账号和子账号默认都可以读写，如果需要自定义权限，请在邮件中注明。
- 如何使用：将要共享的数据放在此文件夹下，就可以在主账号和子账号间共享数据。

课题组之间共享
---------------------------------

- 如何申请：由主账号负责人A向hpc邮箱发送邮件，申请在其主账号下建立 ``acct-expuser01/share`` 文件夹。另外需注明需要开通读权限的其他账号。
- 权限控制：默认只有计费的主账号A能写入，其他账号为只读，如果需要自定义权限，请在邮件中注明。
- 如何使用：将要共享的数据放在此文件夹下，就可以在多个账号间共享数据。
- 如何计费：共享数据的存储费会由主账号A承担。

共享数据到集群外
------------------------

将“交我算”账号的数据共享给非超算用户，可以使用https://scidata.sjtu.edu.cn/提供的服务。通过可视化平台在超算上打开平台网站，上传数据后分享链接至他人。具体操作步骤可以查看文档：

* :ref:`scidatausage`

其他问题
===================

1. 用户可以自行设置共享文件夹的权限吗？

可以，在申请建立共享文件夹之后，主账号负责人可以自行设置共享文件夹的权限，将数据共享给其他账号。以下示例假设主账号用户expuser01在PI的个人目录为 ``/lustre/home/acct-exp/expuser01`` ，思源一号个人目录 ``/dssg/home/acct-exp/expuser01`` ，在其项目组下设立共享文件夹 ``share``。

.. warning::

    添加数据共享权限时，请注意数据安全，谨慎添加权限，尤其是写权限。

.. code:: bash

    # 查看共享文件夹的ACL权限，rwx代表读写权限，r-x代表只读权限
    # 在data, sylogin, sydata等节点访问/dssg存储时
    mmgetacl /dssg/home/acct-exp/share
    # 在data, pilogin, armlogin等节点访问/lustre存储时
    getfacl /lustre/home/acct-exp/share

    # 授予子账号expuser02只读权限，expuser03读写权限
    # 在data, sylogin, sydata等节点访问/dssg存储时，请准备一个文本文件作为指令输入，一次性描述您期望的最终权限配置。
    vim input.txt
        user::rwxc
        group::r-x-
        other::r-x-
        mask::rwxc
        user:expuser02:r-xc
        user:expuser03:rwxc
    mmputacl -i input.txt /dssg/home/acct-exp/share
    # 在data, pilogin, armlogin等节点访问/lustre存储时，您需要逐个添加、删除目标用户的权限。
    setfacl -m u:expuser02:r-x /lustre/home/acct-exp/share
    setfacl -m u:expuser03:rwx /lustre/home/acct-exp/share

2. 如何将共享文件夹中的文件设置成只读？

假设某个账号需要共享一个文件，并且希望这个文件别人无法修改，这时可以将文件权限修改为644，只让文件的所有者能修改，其他人为只读。

.. code:: bash

    # 假设主账号用户expuser01在思源一号个人目录/dssg/home/acct-exp/expuser01，共享文件夹路径为/dssg/home/acct-exp/share，需要共享的文件名为file
    # 查看文件的权限，第三列的用户名代表所有者
    ls -l /dssg/home/acct-exp/share/file

    # 修改文件的权限为644（rw-r--r--），让其他用户只能读
    chmod 644 /dssg/home/acct-exp/share/file

3. 用户可以自行设置家目录下文件的ACL权限来共享数据吗？

以 ``/lustre/home/acct-exp/expuser01/hello.txt`` 为例，其他用户要访问 ``hello.txt`` ，需要有权限进入其上层父目录 ``expuser01`` ，即 ``expuser01`` 的家目录。默认情况下超算平台为用户家目录设置的权限为 ``rwx------`` ，因此仅对 ``hello.txt`` 添加ACL配置还不足以让其他用户访问到此文件。如果确实需要共享家目录中的数据，您还需要对家目录本身添加对象用户的进入权限x，例如：

.. code:: bash

    setfacl -m u:expuser02:r-x /lustre/home/acct-exp/expuser01

**在您决定打开家目录的大门之前，您应当仔细检查家目录中各个文件的权限配置，请确保其他不希望共享的文件没有other域授权，私有数据的权限位不应多于 rwxr-x---**
