QuickPIC
=================

QuickPIC 是基于 UPIC 框架开发的 3D 并行（MPI & OpenMP Hybrid）准静态 PIC 代码。QuickPIC 可以有效地模拟基于等离子体的加速器问题。



编译QuickPIC
-----------------

自行在X86平台上编译QuickPIC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

首先申请计算资源：

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

Hypre库的编译需要OpenMPI、HDF5、JSON-Fortran。请根据自己的需要选择合适的OpenMPI及GCC版本。这里我们选择加载CPU及GPU平台上全局部署的 ``openmpi/3.1.5-gcc-9.2.0``：

.. code:: bash

   $ module purge
   $ module load openmpi/3.1.5-gcc-9.2.0 hdf5/1.10.6-gcc-9.2.0-openmpi json-fortran/6.11.0-gcc-9.2.0

进入QuickPIC的github中clone源代码

.. code:: bash

   $ git clone https://github.com/UCLA-Plasma-Simulation-Group/QuickPIC-OpenSource.git

进入 ``QuickPIC-OpenSource-master/source`` 文件夹并进行编译:

.. code:: bash

   $ cd QuickPIC-OpenSource-master/source
   $ ./configure -prefix=/lustre/home/$YOUR_ACCOUNT/$YOUR_USERNAME/Quick
   $ make

编译完成之后，在家目录下会出现一个 ``qpic.e`` 可执行文件。

.. code:: bash

   $ export OMP_NUM_THREADS=1
   $ mpirun -np 1 ./qpic.e


参考资料
--------
- Hypre主页 https://github.com/UCLA-Plasma-Simulation-Group/QuickPIC-OpenSource

