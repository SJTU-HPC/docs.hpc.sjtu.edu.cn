.. _intel:

Intel Compiler
========================

Intel编译器套件包含以下语言：

.. code::
           
    C、 C++、 Fortran


集群上的Intel编译器
---------------------

+-----------------+-----------------------------+----------+
| 版本            | 加载方式                    | 平台     |
+=================+=============================+==========+
| intel-21.4.0    | module load oneapi/2021.4.0 | 思源一号 |
+-----------------+-----------------------------+----------+
| intel-21.4.0    | module load oneapi/2021.4.0 | pi2.0    |
+-----------------+-----------------------------+----------+

加载oneapi模块后，会同时导入MKL库、intel compiler等。

.. code:: bash

   module load oneapi/2021.4.0
   module list
   Currently Loaded Modules:
   1) intel-oneapi-compilers/2021.4.0   2) intel-oneapi-mpi/2021.4.0   
   3) intel-oneapi-mkl/2021.4.0         4) intel-oneapi-tbb/2021.4.0



常见问题
---------

1. 编译/运行程序时提示“license has expired”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A:** License过期不影响原有软件运行，新编译软件建议使用最新部署的intel编译器套件进行编译 \ ``module load oneapi/2021.4.0``\ 。



参考资料
--------

-  `一篇比较详细的intel程序优化教程 <https://blog.csdn.net/gengshenghong/article/details/7034748/>`__
-  `intel-compiler相关文档 <https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-setup/using-the-command-line/using-compiler-options.html/>`__
