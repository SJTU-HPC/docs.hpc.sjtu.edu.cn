.. _gnu:

GNU Compiler Collection
=========================

GNU(GNU Compiler Collection)，缩写为"GCC"，即"GNU编译器套件"，可将其理解成是多个编译器的集合。
支持的语言包括C、C++、Fortran、Ada、Object-C和Java等。

判断GCC支持的编译语言方法
--------------------------

.. code:: bash

   [hpc@node499 ~]$ gcc --version
   gcc (GCC) 8.3.1 20191121 (Red Hat 8.3.1-5)
   
   [hpc@node499 ~]$ g++ --version
   g++ (GCC) 8.3.1 20191121 (Red Hat 8.3.1-5)
   
   [hpc@node499 ~]$ gfortran --version
   GNU Fortran (GCC) 8.3.1 20191121 (Red Hat 8.3.1-5)

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
gcc-8.3.1  module load gcc/8.3.1
gcc-8.5.0  module load gcc/8.5.0
gcc-9.3.0  module load gcc/9.3.0
gcc-10.3.0 module load gcc/10.3.0
gcc-11.2.0 module load gcc/11.2.0
========== ===============================

思源一号上的GCC默认版本为: 8.3.1。

.. _π2.0上的GCC:

π2.0上的GCC
---------------------

========== ===============================
版本       加载方式
========== ===============================
gcc-5.5.0  module load gcc/5.5.0
gcc-7.4.0  module load gcc/7.4.0
gcc-8.3.0  module load gcc/8.3.0
gcc-9.2.0  module load gcc/9.2.0
gcc-9.3.0  module load gcc/9.3.0
gcc-10.2.0 module load gcc/10.2.0
gcc-11.2.0 module load gcc/11.2.0
========== ===============================

π2.0上的GCC默认版本为: 4.8.5。

.. _ARM上的GCC:

ARM上的GCC
------------------

========== ===============================
版本       加载方式
========== ===============================
gcc-8.4.0  module load gcc/8.4.0
gcc-8.5.0  module load gcc/8.5.0
gcc-9.3.0  module load gcc/9.3.0
========== ===============================

ARM平台上的GCC默认版本为: 4.8.5。

参考资料
--------

-  `Top 20
   licenses <https://web.archive.org/web/20160719043600/>`__
   https://www.blackducksoftware.com/top-open-source-licenses/
