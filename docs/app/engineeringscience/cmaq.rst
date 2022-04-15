.. _cmaq:

CMAQ
====

简介
----

CMAQ（The Community Multiscale Air Quality Modeling System）是美国环境保护局开发的可用于空气质量模型模拟的开源项目，该软件结合了大气科学和空气质量建模方面的现有知识，同时与多处理器计算技术相结合，可有效预测臭氧、颗粒物、有毒物质和酸沉降。

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

可执行文件所在的目录
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   CCTM_v532.exe
   /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/cmaq/CMAQ_Project/CCTM
   
   BCON_v532.exe、ICON_v532.exe、mcip.exe
   /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/cmaq/CMAQ_Project/PREP


运行CMAQ的流程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此处使用CMAQ的核心可执行文件CCTM_v532.exe。

首先拷贝CMAQ_Project到本地
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

   mkdir ~/cmaq
   cd ~/cmaq
   cp -r /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/cmaq/CMAQ_Project ./
   
然后将算例解压到CMAQ_Project下的data目录
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

   cd ~/cmaq/CMAQ_Project/data
   cp -r /dssg/share/sample/cmaq/CMAQv5.3.2_Benchmark_2Day_Input.tar.gz ./
   tar xf CMAQv5.3.2_Benchmark_2Day_Input.tar.gz
   mv CMAQv5.3.2_Benchmark_2Day_Input/* ./
   
接下来根据运行核数修改可执行文件的内部参数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

修改的可执行文件为：~/cmaq/CMAQ_Project/CCTM/scripts/run_cctm_Bench_2016_12SE1.csh

.. code:: bash
    
   @ NPCOL  =  16; @ NPROW =  8
   ### NPCOL*NPROW=运行数据的总核数。比如思源上使用2个节点共128核运行数据，参数配置如上
   set END_DATE   = "2016-07-02"
   ### 运行时间为：7月1日-7月2日
   
执行脚本设置如下
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
| Exec time   | 0:06:41  | 0:05:18   | 0:04:26   |
+-------------+----------+-----------+-----------+

参考资料
--------

-  `CMAQ 官网 <https://www.epa.gov/cmaq>`__
