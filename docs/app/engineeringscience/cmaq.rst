.. _cmaq:

CMAQ
====

简介
----

CMAQ(The Community Multiscale Air Quality Modeling System)是由美国环保署开发的多尺度空气质量模型，用于研究从局部到半球尺度的空气污染。CMAQ模拟关注的空气污染物，包括臭氧、颗粒物（PM）和各种空气毒物，以优化空气质量管理。CMAQ的沉积值用于评估生态系统影响，如空气污染物引起的富营养化和酸化。CMAQ将气象学、排放和化学建模结合起来，以模拟不同大气条件下空气污染物的相关情况。其他类型的模型，包括作物管理和水文模型，可以根据需要和CMAQ模拟相联系，以更全面地模拟环境介质中的污染。

可用版本
--------

+-------+-------+----------+-------------------------------------+
| 版本  | 平台  | 构建方式 | 模块名                              |
+=======+=======+==========+=====================================+
| 5.3.2 | |cpu| | 源码     | cmaq/5.3.2-oneapi-2021.4.0 思源一号 |
+-------+-------+----------+-------------------------------------+

算例获取路径 
---------------

.. code:: bash

   /dssg/share/sample/cmaq/CMAQv5.3.2_Benchmark_2Day_Input.tar.gz
   
集群上的CMAQ
-----------------------

- `思源一号上的CMAQ`_

.. _思源一号上的CMAQ:

思源一号上的CMAQ
---------------------

首先拷贝CMAQ_Project到本地
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   mkdir ~/cmaq
   cd ~/cmaq
   cp -r /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/cmaq/CMAQ_Project ./
   
然后将算例解压到CMAQ_Project下的data目录
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd ~/cmaq/CMAQ_Project/data
   cp -r /dssg/share/sample/cmaq/CMAQv5.3.2_Benchmark_2Day_Input.tar.gz ./
   tar xf CMAQv5.3.2_Benchmark_2Day_Input.tar.gz
   mv CMAQv5.3.2_Benchmark_2Day_Input/* ./
   
接下来根据运行核数修改可执行文件的内部参数（CCTM执行方式）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改的可执行文件为：~/cmaq/CMAQ_Project/CCTM/scripts/run_cctm_Bench_2016_12SE1.csh

.. code:: bash
    
   @ NPCOL  =  16; @ NPROW =  8
   ### NPCOL*NPROW=运行数据的总核数。比如思源上使用2个节点共128核运行数据，参数配置如上
   set END_DATE   = "2016-07-02"
   ### 运行时间为：7月1日-7月2日
   
执行脚本设置如下
~~~~~~~~~~~~~~~~~~~~~~~~~

脚本的位置应在： ~/cmaq/CMAQ_Project/CCTM/scripts/

.. code:: bash

   #!/bin/csh 
   #SBATCH --job-name=cmaq
   #SBATCH --partition=64c512g
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load cmaq
   setenv INPDIR /dssg/home/acct-hpc/hpchgc/data/cmaq/cmaq_test2/CMAQ_Project/data/2016_12SE1 
   csh run_cctm_Bench_2016_12SE1.csh

运行结果(单位为：秒，越低越好)
------------------------------

思源一号上CMAQ的运行时间
~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------------------+
|             cmaq/5.3.2-oneapi-2021.4.0         |
+=============+==========+===========+===========+
| 核数        | 64       | 128       | 256       |
+-------------+----------+-----------+-----------+
| Exec time   | 0:06:41  | 0:04:52   | 0:04:26   |
+-------------+----------+-----------+-----------+
