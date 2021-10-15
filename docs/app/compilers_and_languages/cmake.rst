.. _cmake:

CMAKE
=====

CMake是一个跨平台的编译工具，能够输出各种各样的makefile或者project文件,并不直接构建出最终的软件，而是生成标准的Makefile文件,然后再使用Make进行编译。

CMAKE使用方式如下
------------------------

编写CMakeLists.txt

.. code:: bash

   # CMake 最低版本号要求
   cmake_minimum_required (VERSION 2.8)
   # 项目信息
   project (demo)
   # 指定生成目标
   add_executable(demo demo.cpp)

demo.cpp内容如下

.. code:: bash

   int main()
   {
       return 0;
   }

使用方式如下所示

.. code:: bash

   mkdir build
   cd build
   cmake ../
   make
   ./demo

使用OpenMP示例如下
------------------------

CMakeLists.txt文件如下

.. code:: bash
 
   cmake_minimum_required(VERSION 3.3)
   project(openmp)

   OPTION (USE_OpenMP "Use OpenMP" ON)
   IF(USE_OpenMP)
       FIND_PACKAGE(OpenMP)
   IF(OPENMP_FOUND)
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   ENDIF()
   ENDIF()

   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

   set(SOURCE_FILES hello-omp.c)
   add_executable(hello-omp ${SOURCE_FILES})
   
hello-omp.c如下所示

.. code:: bash

   #include <stdio.h>
   #include <omp.h>
   int main()
   {
   #pragma omp parallel
       {
           int id = omp_get_thread_num();
           printf("hello, from %d.\n", id);
       }
       return 0;
   }

执行如下命令

.. code:: bash

   mkdir build
   cd build
   cmake ../
   make
   ./hello-omp
