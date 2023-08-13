********
快速上手
********

为什么需要交我算HPC+AI平台？
===============================

您可能需要大规模计算，超出笔记本电脑或工作站的处理能力；您可能有太多数据，需要海量存储和高速读写；您可能需要先进高效的 GPU 资源，抑或是大内存节点。这些，都能在交我算HPC+AI平台上实现。


交我算平台面向全校师生提供服务，支撑和催化学校的科研发展。重点支持校内高水平用户的科研，覆盖各学科门类，支撑海洋学、生物医学、航空航天、机械制造、天体物理等领域的科学研究及工程应用，多篇研究发表于 Science、Nature 等高水平期刊上。


交我算HPC+AI平台硬件资源如何？
===============================

“思源一号” 集群总算力6 PFLOPS（每秒千万亿次），是目前国内高校第一的超算集群，TOP500 榜单排名第132位。CPU 采用双路 Intel Xeon ICX Platinum 8358 32 核，主频 2.6GHz，共938个计算节点；GPU采用NVIDIA HGX A100, 共92块GPU卡。计算节点之间使用Mellanox 100 Gbps Infiniband HDR 高速互联，并行存储的聚合存储能力达10 PB。

π 2.0 超算系统双精度浮点数理论性能 2.1 PFLOPS，拥有 658 个双路节点和 1316 颗第二代英特尔至强金牌 6248 处理器，配以英特尔 Omni-Path 架构的 100 Gbps 高速网络互连，以及全闪存的 NVMeLustre 存储系统，体现了强大的计算能力和先进的设计理念。

AI 计算平台是国内高校计算力能最强的人工智能计算平台。该平台由 8 台 DGX-2 组成，每台 DGX-2 配备 16 块 NVIDIA Tesla V100，深度学习张量计算能力可以达到 16 PFLOPS；通过搭载 NVIDIA NVSwitch 创新技术，GPU 间带宽高达 2.4 TB/s。AI 计算平台采用可扩展架构，使得模型的复杂性和规模不受传统架构局限性的限制，从而可以应对众多复杂的人工智能挑战。

ARM 平台基于 ARM 处理器构建，是国内首台基于 ARM 处理器的校级超算。共 100 个计算节点，与 π 2.0集群实现共享登录、共享 Lustre 文件系统和共享 Slurm 作业调度系统。ARM 超算单节点配备 128 核（2.6 GHz）、256 GB 内存（16 通道 DDR4-2933）、240 GB 本地硬盘，节点间采用 IB 高速互联，挂载 Lustre 并行文件系统。


资源如何选择？
=========================

交我算HPC+AI平台采用 Linux 操作系统，配以 Slurm 作业调度系统，所有计算节点资源和存储资源，均可统一调用。

若是大规模的 CPU 作业，可选择 CPU 队列或思源一号64c512g队列，支持万核规模的并行；

若是小规模测试，可选 small 队列或思源一号64c512g队列；

GPU 作业请至 dgx2 队列或思源一号a100队列；

大内存作业可选择 huge 或 192c6t 两种队列。

详情请见：\ `Slurm 作业调度系统 <../job/slurm.html>`__\


使用交我算HPC+AI平台需要什么？
==================================

您只需要一个交我算账号，然后在任何一个常见的浏览器上登录 \ `可视化平台 HPC Studio <../studio/>`__\ ，即可自由使用平台。

可视化平台 HPC Studio 使得登录更便捷，无需安装客户端，大大提升使用体验。浏览器包含电脑端和移动端的 Chrome, Firefox, Edge, Safari 等。

当然您也可以通过 `SSH 客户端 <../login/>`__\ 连接使用平台。

了解平台使用方法，请查看 \ `常见问题 <../faq/>`__\ ;

简短版使用手册（Cheat Sheet）：`交我算平台使用手册简短版 <https://hpc.sjtu.edu.cn/Item/docs/Pi_GetStarted.pdf>`__




如何申请账号？
=========================

交我算平台服务于交大师生。账号申请者需为交大及附属医院在职教师或博士后。

主账号申请
^^^^^^^^^^^^^^^^

在“交我办”（或 `我的数字交大 my.sjtu.edu.cn <https://my.sjtu.edu.cn>`_ ）中的“交我算”里申请。我们将会在两个工作日内开通账号。


子账号申请
^^^^^^^^^^^^^^^^

每个主账号可免费申请子账号。

子账号由账号负责人或子账号使用人在“交我办”（或 `我的数字交大 my.sjtu.edu.cn <https://my.sjtu.edu.cn>`_ ）中的“交我算”里申请。

开通账号之前，还需账号使用者完成“新手上路测试题”，共20题。合格分数为90分，可重复完成，需要将完成后的截图上传。

测试题：https://f.kdocs.cn/ew/3aYIdZR9/

参考视频：https://vshare.sjtu.edu.cn/play/26809b70229c549722c5afd0cf868f77


可参阅的资料：

a) 用户文档：https://docs.hpc.sjtu.edu.cn

b) HPC 网站：https://hpc.sjtu.edu.cn

c) 简短版使用手册（Cheat Sheet）：https://hpc.sjtu.edu.cn/Item/docs/Pi_GetStarted.pdf

参考教学视频
=============

a) `2022 春季用户培训之交我算简介 <https://vshare.sjtu.edu.cn/play/28ce02466c35836b7738fd60ce159289>`_ 

b) `2022 春季用户培训之交我算新手上路 <https://vshare.sjtu.edu.cn/play/8120ee2c8228e693ca78f0190b2e0e24>`_


提交 Hello world 单节点作业
===================================

以单节点的 OpenMP Hello world 为例，演示作业提交过程。

1. 撰写名为 hello_world.c 代码如下

.. code:: c

   #include <omp.h>
   #include <stdio.h>
   #include <stdlib.h>

   int main (int argc, char *argv[])
   {
   int nthreads, tid;

     /* Fork a team of threads giving them their own copies of variables */
     #pragma omp parallel private(nthreads, tid)
       {

        /* Obtain thread number */
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        /* Only master thread does this */
        if (tid == 0)
          {
           nthreads = omp_get_num_threads();
           printf("Number of threads = %d\n", nthreads);
          }

        }  /* All threads join master thread and disband */
   }


2. 使用 GCC 编译

.. code:: bash

   $ module purge
   $ module load gcc
   $ gcc -fopenmp hello_world.c -o hello_world

3. 在本地测试运行 4 线程应用程序

.. code:: bash

   $ export OMP_NUM_THREADS=4 && ./hello_world

4. 编写一个名为 hello_world.slurm 的作业脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=hello_world
   #SBATCH --partition=small
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 8
   #SBATCH --ntasks-per-node=8

   ulimit -l unlimited
   ulimit -s unlimited

   module load gcc

   export OMP_NUM_THREADS=8
   ./hello_world

5. 提交到 SLURM

.. code:: bash

   $ sbatch hello_world.slurm

.. tip:: 编译和作业提交都需要登录到 HPC+AI平台集群，参考本节 `使用交我算HPC+AI平台需要什么？ <https://docs.hpc.sjtu.edu.cn/quickstart/index.html#id5>`_。

登录可视化计算平台
========================

HPC Studio 可视化平台，集成 web shell、文件管理、作业提交、可视化应用等一站式服务。

登录方法：

在浏览器中打开：\ `HPC Studio 可视化平台 <https://studio.hpc.sjtu.edu.cn>`__\

详情请见：\ `HPC Studio 可视化平台使用方法 <../studio/basic.html>`__\
