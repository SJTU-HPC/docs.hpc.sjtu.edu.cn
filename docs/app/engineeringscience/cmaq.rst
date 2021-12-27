.. _cmaq:

CMAQ
====

简介
----
WRF(Weather Research and Forecasting Model)模式是有美国环境预报中心(NCEP),
美国国家大气研究中心（NCAR）以及多个大学、研究所和业务部门联合研发的一种统一的中尺度天气预报模式。
WRF模式适用范围很广，从中小尺度到全球尺度的数值预报和模拟都有广泛的应用。

CPU版WRF(4.3.1)
----------------

WPS前期处理数据所需要的geo包的路径为：

.. code:: bash

   /lustre/opt/contribute/cascadelake/wrf_cmaq/geo 

复制WPS和WRF环境到本地，并导入相应模块

.. code:: bash

   srun -p cpu -n 40 --pty /bin/bash
   cd $HOME
   mkdir wrf
   cd wrf
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/packet_1/WPS-4.3.1 ./
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/packet_1/WRF-master ./

   module load wrf_cmaq/5.3.3-wrf-4.3.1


