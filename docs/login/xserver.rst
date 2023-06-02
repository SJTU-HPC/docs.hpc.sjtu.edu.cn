****************************
使用X Server显示图形界面
****************************

``X11`` 是UNIX上图形视窗系统的标准规范，主要分为3个部分：X Server（X服务器）、X Client（X客户端）、Window Manager（窗口管理器）。X Server是整个X Window System的中心，协调X客户端和窗口管理器的通信。

``X Server`` 运行在本地，推荐 ``MobaXterm`` 终端工具，包含了 ``X server`` 功能。使用前需要确认开启了 ``X11-Forwarding``，并启动 ``X server``。

启动步骤
----------

以思源一号为例：

.. code-block:: bash

   module load rdp                             # 加载远程桌面及VNC启动脚本
   sbatch -p 64c512g -n 4 -J rdp --wrap="rdp"  # 提交计算节点执行
   squeue                                      # 查看分配到的节点
   ssh -X node055                              # 通过登录节点登录到计算节点，需要-X参数
   module load relion                          # 运行GUI程序
   relion

.. image:: /img/rdp.png

.. warning::

   ``X Server`` 显示图形界面的方式需要保持SSH连接，不然GUI程序会中断运行。
