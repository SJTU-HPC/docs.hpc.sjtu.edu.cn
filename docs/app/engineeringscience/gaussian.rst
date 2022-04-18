.. _gaussian:

GAUSSIAN
=============

简介
----

GAUSSIAN 是一个量子化学软件包，它是目前应用最广泛的计算化学软件之一，其代码最初由理论化学家、1998年诺贝尔化学奖得主约翰·波普爵士编写，其名称来自于波普在软件中所使用的高斯型基组。使用高斯型基组是波普为简化计算过程缩短计算时间所引入的一项重要近似。

GAUSSIAN 使用需要授权。请先确认是否拥有使用许可。


GAUSSIAN 安装方法
-----------------------

以下介绍在思源一号上安装 GAUSSIAN。

.. code:: bash

   srun -p 64c512g -N 1 -n 8 --pty /bin/bash     # 申请计算节点编译

   export g16root=/dssg/home/acct-XXX/XXX
   export GAUSS_SCRDIR=/dssg/home/acct-XXX/XXX/tmp
   source /dssg/home/acct-XXX/XXX/g16/bsd/g16.profile

   tar -xjvf G16-A03-AVX2.tbz
   cd g16
   ./bsd/install


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

   module purge
   g16 test.gjf


使用如下指令提交：

.. code:: bash

   $ sbatch test.slurm
