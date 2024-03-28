Rocky 队列使用
==================

Rocky 队列开放测试
------------------

CentOS 8 操作系统已于 2021 年 12 月停止维护，为应对安全风险，计划将思源一号集群的操作系统从 CentOS 8 替换到生命周期健康的 Rocky Linux。

目前提供了部署 Rocky Linux 操作系统的 ``64c512g_rocky`` CPU 节点队列，大部分主流应用已适配，用户可以在队列测试自己的应用。
开放测试时间请关注用户群通知，如在使用测试队列时遇到问题，请发送邮件到 \ `hpc 邮箱 <mailto:hpc@sjtu.edu.cn>`__\ 。

如何使用
-----------------

登录
~~~~~~~~~~~~~~~~~

-  使用 ssh 软件工具登录到 ``sylogin.hpc.sjtu.edu.cn``，跳转到 ``sylogin1``

.. code:: bash

    ssh sylogin1

在 ``64c512g_rocky`` 队列申请CPU资源
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

部署 Rocky Linux 操作系统的 CPU 节点队列为 ``64c512g_rocky``，使用方法和 ``64c512g`` 队列基本一致。
为了尽可能使更多用户有机会测试应用，队列进行了一定的限制：每个主账号下所有子账号能使用的cpu核数≤384核（6节点），作业时间限制为三天。
申请资源时需要指定使用 ``rocky`` 的 qos。

**slurm 脚本**

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --partition=64c512g_rocky
    #SBATCH --qos=rocky
    #SBATCH --nodes 1
    #SBATCH --ntasks-per-node=1

    hostname

**srun 交互式作业**

.. code:: bash

    srun -q rocky -p 64c512g_rocky -n 1 --pty /bin/bash