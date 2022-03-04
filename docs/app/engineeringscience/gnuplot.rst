.. _gnuplot:

gnuplot
==========

简介
-------

Gnuplot 是一款小巧而强大的开源科研绘图工具。绘图质量高且快，易学易用，仅需少量代码，就可得到用于发表的高质量图片，中英文参考文档丰富。Gnuplot可做简单的数据处理和分析，比如统计和多参数函数拟合。

π 集群上的 Gnuplot
---------------------

Gnuplot 需要在 HPC Studio 可视化平台的“远程桌面”里使用。π 集群登录节点不支持 Gnuplot 图形显示。

使用方法
---------------------

先在 HPC Studio 里启动远程桌面，再调用 Gnuplot。

1. 在 HPC Studio 上启动远程桌面

1) 浏览器打开 https://studio.hpc.sjtu.edu.cn

2) 顶栏 ``Interactive Apps`` 下拉菜单，选择第一个 ``Desktop``

3) Desktop 里第一个 ``Desktop Instance Size`` ，选择最基本的 ``1core`` ，然后点击 ``Launch``

4) 等待几秒，甚或更长时间，取决于 small 队列可用资源量。Studio    的远程桌面以一个正常的 small 队列作业启动

5) 启动后，右上角会显示 ``1 node 1 core Running`` . 然后点击 ``Launch Desktop``

6. 在远程桌面里调用 Gnuplot

在远程桌面空白处，右键单击， ``Open Terminal Here`` 打开终端，输入 ``gnuplot`` 命令：

.. code:: bash

   $ gnuplot
   gnuplot> p x    (绘制 y = x 函数)

另一个稍复杂的作图示例：

.. code:: bash

   $ gnuplot
   gnuplot> set title "Iteration within plot command"
            set xrange [0:3]
            set label 1 "plot for [n=2:10] sin(x*n)/n" at graph .95, graph .92 right
            plot for [n=2:10] sin(x*n)/n notitle lw (13-n)/2

.. image:: /img/gnuplot1.*

使用 ``q`` 退出 gnuplot。

绘图操作见动图：

.. image:: /img/gnuplot.*


注意：远程桌面使用完毕后需退出，否则会持续计费。两种退出方法：

1. 在 Studio 界面上点 ``Delete`` 删除该作业。
   
2. 在 集群终端里输入 ``squeue`` 命令查看作业，并用 ``scancel`` 终止该作业。



参考资料
--------

-  gnuplot http://www.gnuplot.info/
