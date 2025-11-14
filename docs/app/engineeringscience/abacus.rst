.. _abacus:

ABACUS
======

简介
----

ABACUS是一个专为大规模电子结构模拟而设计的开源软件包。该软件自2007年以来由中国科学技术大学中国科学院量子信息重点实验室的团队开发。

运行 ABACUS 方法
----------------

π和思源一号集群已经安装好了ABACUS运行环境。要运行模拟，请遵循以下步骤。

1. 申请计算节点

π超算
~~~~~

.. code:: bash

   srun -p cpu -n 4 --pty /bin/bash

思源一号
~~~~~~~~

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash

2. 进入计算节点后，加载模块

.. code:: bash

   module load miniconda3

3. 激活环境

.. code:: bash

   conda init

   source ~/.bashrc

   conda activate /lustre/share/conda_env/abacus_env #在π集群上
   conda activate /dssg/share/conda_env/abacus_env #在思源一号集群上

4. 进入数据目录

.. code:: bash

   cd /path/to/your/data_directory

5. 执行模拟

.. attention::

   严禁在登录节点上直接运行命令。必须通过 Slurm 调度系统将作业提交到计算节点。

.. code:: bash

   OMP_NUM_THREADS=1 mpirun -n 4 abacus

参考资料
--------

-  `ABACUS 官网 <https://abacus.deepmodeling.com/en/latest/quick_start/easy_install.html>`__