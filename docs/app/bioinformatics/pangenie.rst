pangenie
=============

PanGenie是一种输入数据为短读型的基因分型器，通过考虑已知单倍型的信息（通过图中的路径表示）可有效计算泛基因组图中以气泡表示的变异的基因型。

可用版本
----------------------------------------

+---------+--------+----------+----------------------------------------------------+
| 平台    | 版本   | 构建方式 | 路径                                               |
+=========+========+==========+====================================================+
|思源一号 | v2.0.0 | 容器     | /dssg/share/imgs/pangenie/pangenie_v2.0.0.sif      |
+---------+--------+----------+----------------------------------------------------+
| pi2.0   | v2.0.0 | 容器     | /lustre/share/img/x86/pangenie/pangenie_v2.0.0.sif |
+---------+--------+----------+----------------------------------------------------+
| ARM     | v2.0.0 | 容器     | /lustre/share/img/arm/pangenie/pangenie_v2.0.0.sif |
+---------+--------+----------+----------------------------------------------------+

算例路径
---------

+----------+--------------------------------+
| 平台     | 路径                           |
+==========+================================+
| 思源一号 | /dssg/share/sample/pangenie    |
+----------+--------------------------------+
| pi 2.0   | /lustre/share/samples/pangenie |
+----------+--------------------------------+
| ARM      | /lustre/share/samples/pangenie |
+----------+--------------------------------+

使用方法
---------------------

- `思源一号 PanGenie`_

- `π2.0 PanGenie`_

- `ARM PanGenie`_

.. _思源一号 PanGenie:

思源一号 PanGenie
-------------------

首先拷贝数据到本地
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd 
   mkdir pangenie
   cp -r /dssg/share/sample/pangenie/* ./

然后在数据目录下提交如下脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=pangenie_sy
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   IMAGE=/dssg/share/imgs/pangenie/pangenie_v2.0.0.sif
   singularity exec $IMAGE PanGenie -i test-reads.fa -r test-reference.fa -v test-variants.vcf -o test -e 100000 

.. _π2.0 PanGenie:

π2.0 PanGenie
-------------------

首先，拷贝数据到本地
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd 
   mkdir pangenie
   cp -r /lustre/share/samples/pangenie/* ./

然后，在数据目录下提交如下脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=pangenie_x86
   #SBATCH --partition=small
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   IMAGE=/lustre/share/img/x86/pangenie/pangenie_v2.0.0.sif
   singularity exec $IMAGE PanGenie -i test-reads.fa -r test-reference.fa -v test-variants.vcf -o test -e 100000

.. _ARM PanGenie:

ARM PanGenie
-------------------

首先将数据拷贝到本地
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd 
   mkdir pangenie
   cp -r /lustre/share/samples/pangenie/* ./

然后提交如下作业脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=pangenie_arm
   #SBATCH --partition=arm128c256g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   IMAGE=/lustre/share/img/arm/pangenie/pangenie_v2.0.0.sif
   singularity exec $IMAGE PanGenie -i test-reads.fa -r test-reference.fa -v test-variants.vcf -o test -e 100000

PanGenie运行结果
-----------------

PanGenie-思源一号
~~~~~~~~~~~~~~~~~~

.. code:: bash

   ###### Summary ######
   time spent reading input files: 0.532767 sec
   time spent counting kmers:      0.116464 sec
   time spent selecting paths:     0.000311676 sec
   time spent determining unique kmers:    0.00101096 sec
   time spent genotyping chromosome chr1:  0.0527584
   total running time:     0.703721 sec
   total wallclock time: 0.702943 sec
   Total maximum memory usage: 0.024004 GB

PanGenie-pi 2.0
~~~~~~~~~~~~~~~~

.. code:: bash

   ###### Summary ######
   time spent reading input files: 0.0779914 sec
   time spent counting kmers:      0.107329 sec
   time spent selecting paths:     0.00195029 sec
   time spent determining unique kmers:    0.0354376 sec
   time spent genotyping chromosome chr1:  0.0532523
   total running time:     0.276735 sec
   total wallclock time: 0.274615 sec
   Total maximum memory usage: 0.015336 GB

PanGenie-ARM
~~~~~~~~~~~~~~~~

.. code:: bash

   ###### Summary ######
   time spent reading input files: 0.00858966 sec
   time spent counting kmers:      0.0387264 sec
   time spent selecting paths:     0.0010025 sec
   time spent determining unique kmers:    0.00505522 sec
   time spent genotyping chromosome chr1:  0.215358
   total running time:     0.270459 sec
   total wallclock time: 0.268968 sec
   Total maximum memory usage: 0.0128 GB

参考资料
----------------

- PanGenie GitHub https://github.com/eblerjana/pangenie
