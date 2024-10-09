.. _gnu:

GNU Compiler Collection
=========================

GNU(GNU Compiler Collection)，缩写为"GCC"，即"GNU编译器套件"，可将其理解成是多个编译器的集合。
支持的语言包括C、C++、Fortran、Ada、Object-C和Java等。

查看GCC的版本
--------------------------

.. code:: bash

   [hpc@node499 ~]$ gcc --version
   gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22)
   
   [hpc@node499 ~]$ g++ --version
   g++ (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22)
   
   [hpc@node499 ~]$ gfortran --version
   GNU Fortran (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22)

集群平台上的GCC
----------------

- `思源一号上的GCC`_

- `π2.0上的GCC`_

- `ARM上的GCC`_

.. _思源一号上的GCC:

思源一号上的GCC
------------------------

========== ===============================
版本       加载方式
========== ===============================
gcc-8.5.0  module load gcc/8.5.0
gcc-9.3.0  module load gcc/9.3.0
gcc-9.4.0  module load gcc/9.4.0
gcc-10.3.0 module load gcc/10.3.0
gcc-11.2.0 module load gcc/11.2.0
gcc-12.3.0 module load gcc/12.3.0
========== ===============================

思源一号上的GCC默认版本为: 8.5.0。

.. _π2.0上的GCC:

π2.0上的GCC
---------------------

========== ===============================
版本       加载方式
========== ===============================
gcc-9.3.0  module load gcc/9.3.0
gcc-10.2.0 module load gcc/10.2.0
gcc-11.2.0 module load gcc/11.2.0
gcc-12.3.0 module load gcc/12.3.0
gcc-13.2.0 module load gcc/13.2.0
========== ===============================

π2.0上的GCC默认版本为: 8.5.0。

.. _ARM上的GCC:

ARM上的GCC
------------------

ARM平台上的GCC默认版本为: 10.3.1。

参考资料
--------

-  `Top 20
   licenses <https://web.archive.org/web/20160719043600/>`__
   https://www.blackducksoftware.com/top-open-source-licenses/
