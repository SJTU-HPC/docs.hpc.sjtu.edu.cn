.. _gate:

GATE
=======

简介
----
    
    GATE是OpenGATE协作开发的开源软件，致力于医学成像和放射治疗的数值模拟。目前该软件支持模拟发射断层扫描（正电子发射断层扫描 - PET和单光子发射计算机断层扫描 - SPECT）、计算机断层扫描（CT）、光学成像（生物发光和荧光）和放射治疗实验。使用易于学习的宏观机制来配置简单或高度复杂的实验设置，GATE现在在新的医学成像设备的设计、采集协议的优化和图像重建算法的开发和评估中起着关键作用。


可用版本
---------

+------+-------+----------+-----------------------------------------+
| 版本 | 平台  | 构建方式 | 模块名                                  |
+======+=======+==========+=========================================+
| 9.2  | |CPU| | 容器     | gate/9.2-gcc-8.5.0-singularity 思源一号 |
+------+-------+----------+-----------------------------------------+
| 9.2  | |CPU| | 容器     | gate/9.2-gcc-8.5.0-singularity pi2.0    |
+------+-------+----------+-----------------------------------------+

算例
------

.. code:: bash

   思源一号：
   /dssg/share/sample/gate

   pi2.0:
   /lustre/share/samples/gate

集群上的GATE
--------------

- `思源一号 GATE`_

- `π2.0 GATE`_

.. _思源一号 GATE:

一. 思源一号 GATE
---------------------

思源上拷贝数据至本地目录
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd
   mkdir ~/gate
   cp -r /dssg/share/sample/gate/* ./
   
在文件 ``run.sh`` 中更改具体的执行内容
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   $cat run.sh
   
   #!/bin/bash
   /runGate.sh "-a [nb,1000] mac/main.mac"

提交运行脚本
~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=gate_test
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.error
      
   module load gate
   Gate run.sh

.. _π2.0 GATE:

二. π2.0 GATE
---------------------

π2.0上拷贝数据至本地目录
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd
   mkdir ~/gate
   cp -r /dssg/share/sample/gate/* ./
   
更改文件 ``run.sh`` 中具体的执行内容
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   $cat run.sh
   
   #!/bin/bash
   /runGate.sh "-a [nb,1000] mac/main.mac"

π2.0上提交运行脚本
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=gate_test
   #SBATCH --partition=small
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=2
   #SBATCH --output=%j.out
   #SBATCH --error=%j.error
      
   module load gate
   Gate run.sh

执行结果
---------

思源一号上GATE的运行结果
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   tree output/
   output/
   ├── BeamLineEntrance.root
   ├── BeamLineExit.root
   ├── BeamLineMiddle.root
   ├── GlobalBoxEntrance.root
   ├── IDD-proton-Dose-Squared.txt
   ├── IDD-proton-Dose.txt
   ├── IDD-proton-Dose-Uncertainty.txt
   ├── IDD-proton-Edep.txt
   └── stat-proton.txt
   
   0 directories, 9 files   

π2.0上GATE的运行脚本
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   tree output/
   output/
   ├── BeamLineEntrance.root
   ├── BeamLineExit.root
   ├── BeamLineMiddle.root
   ├── GlobalBoxEntrance.root
   ├── IDD-proton-Dose-Squared.txt
   ├── IDD-proton-Dose.txt
   ├── IDD-proton-Dose-Uncertainty.txt
   ├── IDD-proton-Edep.txt
   └── stat-proton.txt
   
   0 directories, 9 files

参考链接：https://opengate.readthedocs.io/en/latest/introduction.html
