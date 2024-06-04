新 64c512g 队列（Rocky 系统）使用
==================================

Rocky 操作系统
------------------

CentOS 8 操作系统已于 2021 年 12 月停止维护，为应对安全风险，计划将思源一号集群的操作系统从 CentOS 8 替换到生命周期健康的 Rocky Linux。

目前提供了部署 Rocky Linux 操作系统的 ``64c512g`` CPU 节点队列，大部分主流应用已适配。
如在使用队列时遇到问题，请发送邮件到 \ `hpc 邮箱 <mailto:hpc@sjtu.edu.cn>`__\ 。

如何使用
-----------------

登录
~~~~~~~~~~~~~~~~~

-  使用 ssh 软件工具登录到

.. code:: bash

    host: sylogin.hpc.sjtu.edu.cn

在 ``64c512g`` 队列申请CPU资源
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

部署 Rocky Linux 操作系统的 CPU 节点队列为 ``64c512g``，使用方法和原 ``64c512g`` 队列基本一致。

**slurm 脚本**

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --partition=64c512g
    #SBATCH --nodes 1
    #SBATCH --ntasks-per-node=1

    hostname

**srun 交互式作业**

.. code:: bash

    srun -p 64c512g -n 1 --pty /bin/bash