********************
线下数据迁移
********************

大规模数据迁移可以通过移动硬盘进行线下迁移，避免网络传输可能面临的线路拥堵问题。

1. 向 `hpc邮箱 <mailto:hpc@sjtu.edu.cn>`__ 发送邮件申请线下迁移服务。需要自备移动硬盘，邮件中请注明需要迁移的数据量、数据类型、数据存储位置以及目标集群等信息。如果数据比较重要，建议到网络信息中心人工递交硬盘。
2. HPC工作人员会根据申请情况安排迁移时间，并提供相关的进度沟通渠道。工作人员将移动硬盘接入数据传输节点后，用户应当按照下文示例的方法进行数据迁移，自行把控迁移进度。中心工作人员仅在必要的情况下协助切换移动硬盘。
3. 如果需要校验数据完整性，建议参考下文在递交硬盘前先计算一遍数据校验码，在迁移完成后在对新数据计算校验码，并进行比对。

线下递交硬盘
======================

1. 请提供完好、可正常读写的移动硬盘，提前做好病毒查杀，因硬盘自身问题导致的数据传输失败、数据损坏或数据无法识别，由用户自行承担。
2. 如果是裸盘，必须附带可以正常使用的移动硬盘盒、配套电源线及usb3.0或更高的传输线；如果是多块外型同样的硬盘，请在硬盘表面贴上编号或区分标记。
3. 邮寄硬盘时，请优先使用顺丰标快，不接受闪送、到付，错误配送的请自行联系快递员退回。数据传输完成且经用户确认后会回寄到付的快递。
4. 若非寄快递形式，请在工作日线下自行送取至网络信息中心。

迁移操作实例
====================

工作人员收到硬盘后会将其接入数据传输节点，用户可以通过ssh登录到数据传输节点进行数据迁移。以下为迁移操作示例：

1. 使用个人账号登录到数据传输节点 ``data.hpc.sjtu.edu.cn`` ，查看移动硬盘挂在情况。示例中硬盘为sde，已有分区sde1，尚未挂载到目录：

.. code:: bash

    [userA@data ~]$ lsblk
    NAME            MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
    sdb               8:16   0   279G  0 disk 
    ├─sdb1            8:17   0     1G  0 part /boot
    ├─sdb2            8:18   0   600M  0 part /boot/efi
    └─sdb3            8:19   0 277.4G  0 part 
    ├─system-root 253:0    0 273.4G  0 lvm  /
    └─system-swap 253:1    0     4G  0 lvm  [SWAP]
    sde               8:64   0  21.8T  0 disk 
    └─sde1            8:65   0  21.8T  0 part 

2. 在工作人员开通挂载权限后，将硬盘挂载到指定路径，示例中指定路径为 ``/mnt/external``：

.. code:: bash

    [userA@data ~]$ sudo mount /dev/sde1 /mnt/external
    [userA@data ~]$ ls /mnt/external
    disk02  logs  nohup.out

3. 创建tmux会话以保持长连接，否则在长时间的传输过程中一旦ssh会话断开就会导致迁移中断。在tmux会话中执行数据迁移命令，示例中使用rsync命令将数据迁移到冷存储中的个人目录。冷存储个人目录路径与用户家目录相似，只需将最顶层路径替换为 ``/vault`` ：

.. code:: bash

    [userA@data ~]$ tmux new -s migration
    # 假设用户家目录是 /lustre/home/acct-example/userA, 则冷存储个人目录为 /vault/home/acct-example/userA
    [userA@data ~]$ rsync -a --no-owner --no-group --no-perms --partial --info=progress2 /mnt/external/ /vault/home/acct-example/userA
    # 迁移过程中会显示进度信息，期间可以按 Ctrl+b d 组合键退出tmux会话
    # 之后可以通过 tmux attach -t migration 命令重新进入会话查看迁移进度

4. 迁移完成后，卸载移动硬盘并通知工作人员进行下一步处理，例如切换下一块移动硬盘：

.. code:: bash

    [userA@data ~]$ sudo umount /mnt/external

5. （可选）如果需要校验数据完整性，可以在迁移完成后计算新数据的校验码，并与递交硬盘前计算的校验码进行比对：

.. code:: bash

    [userA@data ~]$ find /mnt/target/disk01 -type f -print0 | xargs -0 -P 4 -n 100 md5sum | sort -k 2 > /vault/home/acct-example/userA/disk01_md5.txt
    # 假设已有源数据校验表 /mnt/external/disk01_md5.txt，则可以使用 diff 命令进行比对：
    [userA@data ~]$ diff -u /mnt/external/disk01_md5.txt /vault/home/acct-example/userA/disk01_md5.txt

根据数据规模的不同，计算校验码可能会花费较长时间。平台只支持对传输后新数据的校验计算，源数据的校验请在提交迁移申请前于本地自行完成。

迁移完全部移动硬盘数据后，用户应在三个工作日内取走移动硬盘。