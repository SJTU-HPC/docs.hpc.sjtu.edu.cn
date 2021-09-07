.. _tvm:

Apache TVM
=============

简介
----

Apache TVM是一个开源的机器学习编译框架。旨在帮助机器学习工程师在任何硬件平台优化、运行计算任务。

ARM平台上的 TVM
----------------------

ARM 集群上已全局部署了TVM镜像 ``tvm/0.8.0`` 。



提交 TVM 作业
----------------------

首先需要登陆ARM平台的登录节点，查看 ARM 上已编译的软件模块:

.. code:: bash

   module av tvm

调用该模块:

.. code:: bash

   module load tvm/0.8.0

以下 slurm 脚本，在arm队列申请1个cpu核心调用 ``tvm/0.8.0`` 模块。脚本名称可设为 slurm.test

.. code:: bash

   #!/bin/bash
   #SBATCH -J test
   #SBATCH -p arm128c256g
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 1

   module load tvm/0.8.0

   python -c "import tvm; print(tvm.__version__);"

使用如下指令提交：

.. code:: bash

   $ sbatch slurm.test

参考资料
--------

- `Apache TVM 官网 <https://tvm.apache.org/>`__

