.. _cmaq:

CMAQ
====

简介
-----

CMAQ(The Community Multiscale Air Quality Modeling System)是由美国环保署开发的多尺度空气质量模型，用于研究从局部到半球尺度的空气污染。CMAQ模拟关注的空气污染物，包括臭氧、颗粒物（PM）和各种空气毒物，以优化空气质量管理。CMAQ的沉积值用于评估生态系统影响，如空气污染物引起的富营养化和酸化。CMAQ将气象学、排放和化学建模结合起来，以模拟不同大气条件下空气污染物的相关情况。其他类型的模型，包括作物管理和水文模型，可以根据需要和CMAQ模拟相联系，以更全面地模拟环境介质中的污染。

调用方式
------------------------------------------------

为方便数据处理，可拷贝CMAQ中的文件到本地

.. code:: bash

   srun -p small -n 2 --pty /bin/bash

   cd $HOME
   mkdir wrf_cmaq
   cd wrf_cmaq
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/install_1/CMAQ ./
   module load wrf_cmaq/5.3.3-wrf-4.3.1
