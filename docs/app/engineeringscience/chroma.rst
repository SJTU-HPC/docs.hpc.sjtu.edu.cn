.. _chroma:

CHROMA
=======

简介
----

CHROMA是一款格点量子色动力学（LQCD）数值模拟软件包，它是一个为解决夸克和胶子理论而设计的物理应用程序，其属于美国USQCD合作组在美国SciDac经费的支持下开发的USQCD软件集中的子模块。Chroma整合了QCD基本线性代数操作、多线程/进程QCD信息传递、QCD文件IO、稀疏矩阵求解等众多模块，并通过XML交互协议提供人机接口。

可用的版本
----------

+--------+-------+----------+----------------------------------------------------------+
| 版本   | 平台  | 构建方式 | 镜像路径                                                 |
+========+=======+==========+==========================================================+
| 2021.4 | |cpu| | 容器     | /dssg/share/imgs/chroma/chroma2021.04.sif 思源一号       |
+--------+-------+----------+----------------------------------------------------------+
| 2021.4 | |cpu| | 容器     | /lustre/share/img/chroma/chroma2021.04.sif pi2.0         |
+--------+-------+----------+----------------------------------------------------------+

算例路径
---------

.. code:: bash

   思源一号
   /dssg/share/sample/chroma/szscl_bench.zip

   pi2.0
   /lustre/share/samples/chroma/szscl_bench.zip

软件下载
---------

本文档使用的是在NVIDIA GPU云（NGC）上预好的Chroma-2021.04镜像。更多信息请访问

.. code:: bash

   https://catalog.ngc.nvidia.com/orgs/hpc/containers/chroma

使用方法
----------------

- `一. 思源一号 Chroma`_

- `二. π2.0 Chroma`_

.. _一. 思源一号 Chroma:

一. 思源一号 Chroma
--------------------

运行脚本

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=chroma
   #SBATCH --partition=a100
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=24
   #SBATCH --gres=gpu:2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   export QUDA_RESOURCE_PATH=$PWD
   export GPU_COUNT=2
   
   singularity run --nv /dssg/share/imgs/chroma/chroma2021.04.sif mpirun --allow-run-as-root -x ${QUDA_RESOURCE_PATH} -n ${GPU_COUNT} chroma -i ./test.ini.xml -geom 1 1 1 ${GPU_COUNT} -ptxdb ./qdpdb -gpudirect

.. _π2.0 Chroma:

二. π2.0 Chroma
------------------

运行脚本

.. code:: bash

   #!/bin/bash
   
   #SBATCH --job-name=chroma
   #SBATCH --partition=dgx2
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --gres=gpu:2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   export QUDA_RESOURCE_PATH=$PWD
   export GPU_COUNT=2
   
   singularity run --nv /lustre/share/img/chroma/chroma2021.04.sif mpirun --allow-run-as-root -x ${QUDA_RESOURCE_PATH} -n ${GPU_COUNT} chroma -i ./test.ini.xml -geom 1 1 1 ${GPU_COUNT} -ptxdb ./qdpdb -gpudirect


自动编译
--------------------

1.申请计算节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   srun -p 64c512g -n 4 --pty /bin/bash

2.拉取远端镜像
~~~~~~~~~~~~~~~~
            
参考文档：
``https://docs.hpc.sjtu.edu.cn/container/index.html``

.. code:: bash

   singularity pull chroma2021.04.sif docker://nvcr.io/hpc/chroma:2021.04

运行结果如下所示(单位：s，越低越好)
-----------------------------------------

1.Chroma 思源一号
~~~~~~~~~~~~~~~~~~

+---------+----------+
| GPU卡数 | 计算时间 |
+=========+==========+
| 1A100   | 154.2    |
+---------+----------+
| 2A100s  | 22       |
+---------+----------+

2.Chroma π2.0
~~~~~~~~~~~~~~~~

+---------+----------+
| GPU卡数 | 计算时间 |
+=========+==========+
| 1V100   | 258      |
+---------+----------+
| 2V100s  | 40       |
+---------+----------+


参考资料
--------

- Chroma https://jeffersonlab.github.io/chroma
Creating a new branch is quick.
