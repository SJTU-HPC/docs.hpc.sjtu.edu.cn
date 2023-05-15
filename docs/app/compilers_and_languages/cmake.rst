.. _cmake:

CMake
====================

简介
--------

CMake是一个跨平台的编译工具，能够输出各种各样的makefile或者project文件,并不直接构建出最终的软件，而是生成标准的Makefile文件,然后再使用Make进行编译。



CMake使用说明
-----------------------------

在思源一号上使用CMake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 首先在自己的家目录下新建一个目录作为测试目录,并进入该目录：

.. code::

    mkdir cmaketest
    cd  cmaketest


2. 申请计算资源并加载所需模块：

.. code::

    srun -p 64c512g -n 10 --pty /bin/bash

    module purge
    module load openmpi/4.1.1-gcc-11.2.0 
    module load gcc/11.2.0
    module load cmake/3.17.1-gcc-11.2.0


3. 在当前目录下编写如下测试文件cmaketest.cpp：

.. code::

    #include "mpi.h"
    #include <iostream>
    using namespace std;

    int say_hello(int argc, char **argv)
    {
        int myid, numprocs;
        int namelen;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Get_processor_name(processor_name, &namelen);
        cout << "Hello World! Process " << myid << " of " << numprocs << " on " << processor_name << "\n";
        MPI_Finalize();

        return 0;
    }

    int main(int argc, char **argv)
    {
        say_hello(argc, argv);

        return 0;
    }


4. 在当前目录下编写如下CMakeLists.txt文件：

.. code::

    cmake_minimum_required(VERSION 3.15)

    message(STATUS "The CMAKE_VERSION is ${CMAKE_VERSION}.")

    project(cmaketest)

    find_package(MPI REQUIRED)

    message(STATUS "PROJECT_NAME is ${PROJECT_NAME}")

    include_directories ("${MPI_CXX_INCLUDE_DIRS}")

    add_executable(${PROJECT_NAME} cmaketest.cpp )

    target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})



5. 执行以下命令编译源文件。如果编译成功，在build目录下会生成cmaketest可执行文件：

.. code::

  mkdir build
  cd build
  cmake ../
  make


6. 调用MPI运行可执行文件:

.. code::

  mpirun -np 10 ./cmaketest


7. 此时可以在终端看到如下输出结果：

.. code::

    Hello World! Process 1 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 5 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 6 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 7 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 8 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 9 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 0 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 2 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 4 of 10 on node466.pi.sjtu.edu.cn
    Hello World! Process 3 of 10 on node466.pi.sjtu.edu.cn





在pi2.0上使用CMake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；

2. 申请计算资源并加载所需模块：

.. code::

  srun -p cpu -N 1 --ntasks-per-node 40    --pty /bin/bash

  module purge
  module load gcc/9.2.0
  module load cmake/3.18.4-gcc-9.2.0
  module load openmpi/3.1.5-gcc-9.2.0


3. 此步骤和上文完全相同；

4. 此步骤和上文完全相同；

5. 此步骤和上文完全相同；

6. 此步骤和上文完全相同；

7. 此步骤和上文完全相同；




参考资料
----------------

-  `CMake官网 <https://cmake.org/>`__






















