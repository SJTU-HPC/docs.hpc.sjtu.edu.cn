.. _thirdorder:

thirdorder
===========

介绍
-------------

Thirdorder可帮助创建用于计算非谐原子间力常数的输入文件，利用系统的对称性来最小化所需的DFT计算次数，同时该应用也可以允许用户根据运行的结果构建三阶IFC矩阵。

算例获取
---------

思源一号
>>>>>>>>>>>>

.. code:: bash

   /dssg/share/sample/thirdorder
   
π2.0
>>>>>>>>>

.. code:: bash

   /lustre/share/samples/thirdorder

集群上的thirdorder
---------------------

- `思源一号上的thirdorder`_
- `π2.0上的thirdorder`_

.. _思源一号上的thirdorder:


思源一号上的thirdorder
-----------------------------

使用方法如下

.. code:: bash
        
   mkdir ~/thirdorder && cd ~/thirdorder
   cp -r /dssg/share/sample/thirdorder/* ./
   singularity shell /dssg/share/imgs/thirdorder/thirdorder.sif
   thirdorder_espresso.py scf.in

运行结果如下将显示thirdorder能够正常使用

.. code:: bash

   Usage:
        /opt/software/install/sousaw-thirdorder-6050bebe4dd1/thirdorder_espresso.py unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
        /opt/software/install/sousaw-thirdorder-6050bebe4dd1/thirdorder_espresso.py unitcell.in reap na nb nc cutoff[nm/-integer]
                
.. _π2.0上的thirdorder:


π2.0上的thirdorder
-----------------------------

使用方法如下

.. code:: bash
        
   mkdir ~/thirdorder && cd ~/thirdorder
   cp -r /lustre/share/samples/thirdorder/* ./
   singularity shell /lustre/share/img/x86/thirdorder/thirdorder.sif
   thirdorder_espresso.py scf.in

运行结果如下将显示thirdorder能够正常使用

.. code:: bash

   Usage:
        /opt/software/install/sousaw-thirdorder-6050bebe4dd1/thirdorder_espresso.py unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
        /opt/software/install/sousaw-thirdorder-6050bebe4dd1/thirdorder_espresso.py unitcell.in reap na nb nc cutoff[nm/-integer] 

参考链接
----------

-  `thirdorder website <https://bitbucket.org/sousaw/thirdorder/src/master/>`__
