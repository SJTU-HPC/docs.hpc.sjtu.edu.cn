.. _gnuplot:

gnuplot
=======

简介
----

Gnuplot 是一款小巧而强大的开源科研绘图工具。绘图质量高且快，易学易用，仅需少量代码，就可得到用于发表的高质量图片，中英文参考文档丰富。Gnuplot可做简单的数据处理和分析，比如统计和多参数函数拟合。

π 集群上的 gnuplot
---------------------

Gnuplot 需要在 HPC Studio 可视化平台的“远程桌面”里使用。π 集群登录节点不支持 Gnuplot 图形显示。

HPC Studio 可视化平台通过浏览器访问：https://studio.hpc.sjtu.edu.cn

使用 gnuplot
------------

在 HPC Studio 上连接远程桌面
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 浏览器打开 https://studio.hpc.sjtu.edu.cn

2. 顶栏 Interactive Apps 下拉菜单，选择第一个 Desktop

3. Desktop 里第一个 “Desktop Instance Size” 选择最基本的
   1core-desktop（画图所需资源少，1 core 够用），然后点击 Launch

4. 等待几秒，甚或更长时间，取决于 small 队列可用资源量。Studio
   的远程桌面以一个正常的 small 队列作业启动

5. 启动后，右上角会显示 1 node 1 core Running. 然后点击 Launch Desktop

远程桌面启动 gnuplot
~~~~~~~~~~~~~~~~~~~~

在远程桌面空白处右键单击，Open Terminal Here 打开终端

.. code:: bash

   $ gnuplot
   gnuplot> p x    (以绘制 y = x 函数为例)

另一个稍复杂的官方 demo 示例：

.. code:: bash

   $ gnuplot
   gnuplot> set title "Iteration within plot command"
            set xrange [0:3]
            set label 1 "plot for [n=2:10] sin(x*n)/n" at graph .95, graph .92 right
            plot for [n=2:10] sin(x*n)/n notitle lw (13-n)/2

.. image:: /img/gnuplot1.*

结束后退出远程桌面

绘图操作见动图：

.. image:: /img/gnuplot.*


~~~~~~~~~~~~~~~~~~

远程桌面作业，使用完毕后需退出，否则会持续计费。两种退出方法：

1. 在 Studio 界面上点 “Delete” 删除该作业

2. 或在 π 集群上用 squeue 查看作业，并用 scancel 终止该作业



参考资料
--------

-  gnuplot http://www.gnuplot.info/
