*******
bisheng
*******

.. TODO: Guangchao

毕昇编译器是针对鲲鹏平台的高性能编译器。它基于开源LLVM开发，并进行了优化和改进，同时将Flang作为默认的Fortran语言前端编译器。

bisheng编译器使用方式
---------------------

-  首先一定要用 \ ``ssh``\ 登录ARM节点

.. code:: bash

   $ ssh -p 18022 username@202.120.58.248

-  使用 \ ``module``\ 导入应用命令

.. code:: bash

   $ module load bisheng/1.3.1-gcc-9.3.0

-  毕昇编译器使用举例(编译运行hello.c)

.. code:: bash

    #include <stdio.h>
    int main(){
        printf("hello world");
        return 0;
    }

    clang hello.c -o hello.o

    ./hello.o

参考资料
========
