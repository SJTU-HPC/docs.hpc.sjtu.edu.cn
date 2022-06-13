.. _gaussian:

GAUSSIAN
=============

简介
----

GAUSSIAN 是一个量子化学软件包，它是目前应用最广泛的计算化学软件之一，其代码最初由理论化学家、1998年诺贝尔化学奖得主约翰·波普爵士编写，其名称来自于波普在软件中所使用的高斯型基组。使用高斯型基组是波普为简化计算过程缩短计算时间所引入的一项重要近似。

超算平台不提供 GAUSSIAN 程序，也不解答与 GAUSSIAN 授权相关的问题。请用户前往官方网站 https://gaussian.com 咨询软件授权事宜。由于使用非法授权软件产生的后果，由用户承担。


作业脚本示例（假设作业脚本名为 `test.slurm` ）：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g
   #SBATCH -N 1
   #SBATCH -n 8                    # 核数 8
   #SBATCH --ntasks-per-node=8     # 每节点核数
   #SBATCH --output=test.out
   #SBATCH --error=%j.err

   g16 test.gjf


使用如下指令提交：

.. code:: bash

   $ sbatch test.slurm
