OpenBLAS
========

OpenBLAS是一个开源的线性代数库，高效实现了BLAS(Basic Linear Algebra Subprograms)和LAPACK(Linear Algebra PACKage)接口定义的函数，超算平台提供了X86和ARM适用的版本。

可用OpenBLAS版本
----------------

+-------+-------+----------+-------------------------------+
| 版本  | 平台  | 构建方式 | 模块名                        |
+=======+=======+==========+===============================+
| 0.3.7 | |cpu| | Spack    | openblas/0.3.7-gcc-9.2.0      |
+-------+-------+----------+-------------------------------+
| 0.3.7 | |arm| | Spack    | openblas/0.3.7-gcc-9.3.0      |
+-------+-------+----------+-------------------------------+

链接OpenBLAS库
--------------

.. caution:: CPU平台(cpu, small, huge, 192c6t等队列)与ARM平台(arm128c256g队列)指令集不兼容，请勿混用两个平台的二进制程序。

.. tip:: 请使用与OpenBLAS库相匹配的特定版本编译器以获得最佳性能。


下面将分别展示如何在CPU(X86)和ARM平台构建示例程序 ``sampleblas`` ，这个程序调用OpenBLAS提供的 ``cblas_dgemm`` 函数，完成矩阵乘加操作。

sampleblas的源代码 ``sampleblas.c`` 内容如下：

.. code:: c

   #include <cblas.h>
   #include <stdio.h>
   
   int main() {
   
       int i = 0;
       double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};   // A(3x2)
       double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};   // B(2x3)
       double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};  // C(3x3)
   
       const int M = 3; // row of A and C
       const int N = 3; // col of B and C
       const int K = 2; // col of A and row of B
   
       const double alpha = 1.0;
       const double beta = 0.1;
   
       // C = alpha * A * B + beta * C
       cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
   
       for (i = 0; i < 9; i++) {
           printf("%lf ", C[i]);
       }
       printf("\n");

       return 0;
   }

在CPU平台上链接OpenBLAS库
~~~~~~~~~~~~~~~~~~~~~~~~~

在这个示例中我们使用 ``openblas/0.3.7-gcc-9.2.0`` 模块，这个模块使用GCC 9.2.0构建，需要载入OpenBLAS以及与之相匹配的编译器：

.. code:: bash

   $ module load openblas/0.3.7-gcc-9.2.0 gcc/9.2.0-gcc-4.8.5

编译源代码并链接至OpenBLAS库，由于模块中已经预置了头文件、静态库和动态库的路径，因此不需要在命令行中显式制定这些路径：

.. code:: bash

   $ gcc -o sampleblas sampleblas.c -lopenblas

检查二进制程序的动态链接情况，确认已经链接正确的OpenBLAS库：

.. code:: bash

   $ ldd sampleblas
        linux-vdso.so.1 =>  (0x00007fff6bfca000)
        libopenblas.so.0 => /lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/openblas-0.3.7-kf4td3bj4liyg3magigle6h5dubwsrrg/lib/libopenblas.so.0 (0x00002ae926124000)
        libc.so.6 => /lib64/libc.so.6 (0x00002ae926fdc000)
        libm.so.6 => /lib64/libm.so.6 (0x00002ae9273aa000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00002ae9276ac000)
        libgfortran.so.5 => /lustre/opt/cascadelake/linux-centos7-x86_64/gcc-4.8.5/gcc-9.2.0-wqdecm4rkyyhejagxwmnabt6lscgm45d/lib64/libgfortran.so.5 (0x00002ae9278c8000)
        libgomp.so.1 => /lustre/opt/cascadelake/linux-centos7-x86_64/gcc-4.8.5/gcc-9.2.0-wqdecm4rkyyhejagxwmnabt6lscgm45d/lib64/libgomp.so.1 (0x00002ae927d57000)
        /lib64/ld-linux-x86-64.so.2 => /lib/ld-linux.so.2 (0x00002ae925f00000)
        libquadmath.so.0 => /lustre/opt/cascadelake/linux-centos7-x86_64/gcc-4.8.5/gcc-9.2.0-wqdecm4rkyyhejagxwmnabt6lscgm45d/lib64/libquadmath.so.0 (0x00002ae927f8d000)
        libgcc_s.so.1 => /lustre/opt/cascadelake/linux-centos7-x86_64/gcc-4.8.5/gcc-9.2.0-wqdecm4rkyyhejagxwmnabt6lscgm45d/lib64/libgcc_s.so.1 (0x00002ae9281d4000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00002ae9283ec000)
   
这个程序运行时间很短：

.. code:: bash

   $ time ./sampleblas
   -4.950000 10.050000 -0.950000 10.050000 -9.950000 4.050000 7.050000 4.050000 5.050000 
   ./sampleblas  0.00s user 0.01s system 27% cpu 0.045 total 

运行时间更长、消耗时间更多的计算程序，需要编写作业脚本，提交到作业调度系统。

在ARM平台上链接OpenBLAS库
~~~~~~~~~~~~~~~~~~~~~~~~~

在这个示例中我们使用 ``openblas/0.3.7-gcc-9.3.0`` 模块，这个模块使用GCC 9.3.0构建，需要载入OpenBLAS以及与之相匹配的编译器：

.. code:: bash

   $ module load openblas/0.3.7-gcc-9.3.0 gcc/9.3.0-gcc-4.8.5

编译源代码并链接至OpenBLAS库，由于模块中已经预置了头文件、静态库和动态库的路径，因此不需要在命令行中显式制定这些路径：

.. code:: bash

   $ gcc -o sampleblas sampleblas.c -lopenblas

检查二进制程序的动态链接情况，确认已经链接正确的OpenBLAS库：

.. code:: bash

   $ ldd sampleblas
        linux-vdso.so.1 =>  (0x000040002e9c0000)
        libopenblas.so.0 => /lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-9.3.0/openblas-0.3.7-jbipn2oklioz3ym7ra4vh3do3ph5ocou/lib/libopenblas.so.0 (0x000040002e9d0000)
        libc.so.6 => /lib64/libc.so.6 (0x000040002f630000)
        libm.so.6 => /lib64/libm.so.6 (0x000040002f7c0000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x000040002f880000)
        libgfortran.so.5 => /lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-4.8.5/gcc-9.3.0-a5tvx33on7quyl7o2sygvyjqnysfcw6n/lib64/libgfortran.so.5 (0x000040002f8c0000)
        libgomp.so.1 => /lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-4.8.5/gcc-9.3.0-a5tvx33on7quyl7o2sygvyjqnysfcw6n/lib64/libgomp.so.1 (0x000040002fa30000)
        /lib/ld-linux-aarch64.so.1 (0x000040002e970000)
        libgcc_s.so.1 => /lustre/opt/kunpeng920/linux-centos7-aarch64/gcc-4.8.5/gcc-9.3.0-a5tvx33on7quyl7o2sygvyjqnysfcw6n/lib64/libgcc_s.so.1 (0x000040002fa90000)
        libdl.so.2 => /lib64/libdl.so.2 (0x000040002fad0000)

这个程序运行时间很短：

.. code:: bash

   $ time ./sampleblas
   -4.950000 10.050000 -0.950000 10.050000 -9.950000 4.050000 7.050000 4.050000 5.050000 
   ./sampleblas  0.00s user 0.00s system 41% cpu 0.009 total

运行时间更长、消耗时间更多的计算程序，需要编写作业脚本，提交到作业调度系统。

提交依赖OpenBLAS库的作业
------------------------

.. caution:: CPU平台(cpu, small, huge, 192c6t等队列)与ARM平台(arm128c256g队列)指令集不兼容，请勿混用两个平台的二进制程序。

作业成功运行的关键，是加载程序所依赖的软件模块。以 ``sampleblas`` 程序为例，它依赖GCC和OpenBLAS，因此需要在作业脚本中载入相应模块。
此外，OpenBLAS采用OpenMP多线程并行，在作业脚本中为环境变量 ``NUM_OMP_THREADS`` 设置合理数值能达到最佳运行效果。

在CPU平台提交依赖OpenBLAS库的作业
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

准备作业脚本 ``sampleblas.slurm`` ，内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openblas       # 作业名
   #SBATCH --partition=cpu           # cpu队列
   #SBATCH --ntasks-per-node=40      # 每节点核数
   #SBATCH -n 40                     # 作业核心数40(一个节点)
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load openblas/0.3.7-gcc-9.2.0 gcc/9.2.0-gcc-4.8.5

   export NUM_OMP_THREADS=40

   time ./sampleblas

使用 ``sbatch`` 提交作业：

.. code:: bash

   $ sbatch sampleblas.slurm

在ARM平台提交依赖OpenBLAS库的作业
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

准备作业脚本 ``sampleblas.slurm`` ，内容如下：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=openblas       # 作业名
   #SBATCH --partition=arm128c256g   # ARM队列(arm128c256g)
   #SBATCH --ntasks-per-node=128     # 每节点核数
   #SBATCH -n 128                    # 作业核心数128(一个节点)
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   ulimit -s unlimited
   ulimit -l unlimited

   module load openblas/0.3.7-gcc-9.3.0 gcc/9.3.0-gcc-4.8.5

   export NUM_OMP_THREADS=128

   time ./sampleblas

使用 ``sbatch`` 提交作业：

.. code:: bash

   $ sbatch sampleblas.slurm

参考资料
--------

- OpenBLAS官方网站 https://www.openblas.net
