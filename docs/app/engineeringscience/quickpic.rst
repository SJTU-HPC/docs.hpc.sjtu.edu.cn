.. _QuickPIC:

QuickPIC
=================

QuickPIC 是基于 UPIC 框架开发的 3D 并行（MPI & OpenMP Hybrid）准静态 PIC 代码。QuickPIC 可以有效地模拟基于等离子体的加速器问题。

运行QuickPIC的方式
--------------------

申请计算节点

.. code:: bash
 
   salloc -p small -n 4 /bin/bash
   ssh cas*

运行命令如下:    

.. code:: bash

   module load gcc/9.3.0-gcc-4.8.5
   module load openmpi/3.1.5-gcc-9.3.0
   export PATH=$PATH:/lustre/opt/contribute/cascadelake/quickpic/install/HDF5/bin:/lustre/opt/contribute/cascadelake/quickpic/packet/QuickPIC-OpenSource/source
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/opt/contribute/cascadelake/quickpic/install/SZIP/lib:/lustre/opt/contribute/cascadelake/quickpic/install/ZLIB/lib:/lustre/opt/contribute/cascadelake/quickpic/install/HDF5/lib:/lustre/opt/contribute/cascadelake/quickpic/install/json/jsonfortran-gnu-6.10.0/lib

   mkdir ~/quickpic
   cd ~/quickpic
   cp -r /lustre/opt/contribute/cascadelake/quickpic/packet/QuickPIC-OpenSource/source ./
   cd source 

   export OMP_NUM_THREADS=1
   mpirun -np 2 ./qpic.e
