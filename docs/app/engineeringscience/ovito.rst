.. _ovito:

OVITO
=====

简介
----

OVITO 是一款专业实用、功能强大的原子分子可视化及分析软件。界面美观，功能齐全，操作简单，支持超大规模原子快速显示。LAMMPS 的 dump 构型、VASP 的 POSCAR， XDATCAR 等构型均可由 OVITO 查看和编辑。

π 集群上使用 OVITO
---------------------

使用 OVITO 查看原子构型有两种方法：

* 方法一：Studio 可视化平台里直接使用 OVITO（适合快速查看）

* 方法二：本地电脑 OVITO 远程调用集群上的原子构型文件

供测试的原子构型文件： ``cnt.dump`` （388 原子的碳纳米管）

.. code:: bash

   /lustre/archive/samples/ovito/cnt.dump


方法一：Studio 平台里直接使用 OVITO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

特点：无需安装 OVITO、按机时计费。

OVITO 需要在 HPC Studio 可视化平台上使用。集群登录节点不支持 OVITO 可视化显示

1. 浏览器打开 https://studio.hpc.sjtu.edu.cn
2. 顶栏 ``Interactive Apps`` 下拉菜单，选择 ``ovito``
3. 接下来进入资源选择界面。第一个 ``Desktop Instance Size`` 默认选择 ``1core``，点击 ``Launch``
4. 等待几秒，甚或更长时间，取决于 small 队列可用资源量。Studio 上的应用以一个 small 队列作业申请资源
5. 待资源申请成功，新的界面下方会出现 ``Launch ovito``，点击进入 OVITO

注意：OVITO 使用完毕后需终止应用资源，否则之前申请的 small 队列会持续计费。有两种退出方法：

1. 在 Studio 界面上点 ``Delete`` 删除该作业。
   
2. 在 集群终端里输入 ``squeue`` 命令查看作业，并用 ``scancel`` 终止该作业。

方法二：本地 OVITO 调用远程构型文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

特点：速度快、画质无损失、不计费。

在本地电脑开启 OVITO 软件，点击 OVITO 顶上的 ``File`` -> ``Load Remote File``

地址栏按下方格式给定绝对路径。注意：集群路径 ``/lustre`` 前面无冒号

.. code:: bash

   sftp://userXXX@login.hpc.sjtu.edu.cn/lustre/archive/samples/ovito/cnt.dump

.. image:: /img/ovito2.*




参考资料
--------

-  OVITO 官网 http://ovito.org
