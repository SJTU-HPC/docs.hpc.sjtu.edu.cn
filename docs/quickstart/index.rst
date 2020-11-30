********
快速上手
********

为什么需要 π 集群？
=========================

您可能需要大规模计算，超出笔记本电脑或工作站的处理能力；您可能有太多数据，需要海量存储和高速读写；您可能需要先进高效的 GPU 资源，抑或是大内存节点。这些，都能在交大 π集群上实现。


π 集群面向全校师生提供服务，支撑和催化学校的科研发展。重点支持校内高水平用户的科研，覆盖各学科门类，支撑海洋学、生物医学、航空航天、机械制造、天体物理等领域的科学研究及工程应用，多篇研究发表于 Science、Nature 等高水平期刊上。


π 集群硬件资源如何？
=========================

π 集群自 2019 年底上线，双精度浮点数理论性能 2.1 PFlops，拥有 658 台双路节点和 1316 颗第二代英特尔至强金牌 6248 处理器，配以英特尔 Omni-Path 架构的 100 Gbps 高速网络互连，以及全闪存的 NVMeLustre 存储系统，体现强大的计算能力和先进的设计理念。

π 集群上的 AI 计算平台是国内高校计算力能最强的人工智能计算平台。该平台由 8 台 DGX-2 组成，每台 DGX-2 配备 16 块 NVIDIA Tesla V100，深度学习张量计算能力可以达到 16 PFLOPS；通过搭载 NVIDIA NVSwitch 创新技术，GPU 间带宽高达 2.4 TB/s。AI 计算平台采用可扩展架构，使得模型的复杂性和规模不受传统架构局限性的限制，从而可以应对众多复杂的人工智能挑战。


资源如何选择？
=========================

π 集群采用 CentOS 的操作系统，配以 Slurm 作业调度系统，所有计算节点资源和存储资源，均可统一调用。

若是大规模的 CPU 作业，可选择 CPU 队列，支持万核规模的并行；

若是小规模测试，可选 small 队列；

GPU 作业请至 dgx2 队列；

大内存作业可选择 huge 或 192c6t 两种队列。

详情请见：\ `Slurm 作业调度系统 <../job/slurm.html>`__\ 


使用 π 集群需要什么？
=========================

您需要准备好以下内容，使用 π集群：

1. \ `π 集群账号 <../accounts/apply.html>`__\ ;

2. \ `SSH 客户端 <../login/ssh.html>`__\ 或浏览器里使用 \ `可视化平台 HPC Studio <../studio/index.html>`__\;

3. 了解 π 集群使用方法，查看 \ `常见问题 <../faq/index.html>`__\ ;


如何申请帐号？
=========================

π 集群服务于交大师生。账号申请者需为交大在职教职工（包含博士后）。每个账号可免费申请四个子账号。

主账号申请
^^^^^^^^^^^^^^^^

请先阅读
`网络信息中心高性能计算服务 <https://net.sjtu.edu.cn/wlfw/gxnjsfw.htm>`__
，填写《上海交通大学高性能计算申请表》，使用交大邮箱，将申请表发送至 `hpc
邮箱 <mailto:hpc@sjtu.edu.cn>`__\。我们将会在两个工作日内开通账号。


子账号申请
^^^^^^^^^^^^^^^^

每个主账号可免费申请四个子账号。超出四个，将收取 200 元/年/个 的管理费用。

申请子账号，可由账号负责人发邮件申请，也可由子账号使用者使用交大邮箱申请，并抄送账号负责人。

开通子账号之前，还需子账号使用者完成“新手上路”考核（B 站互动视频），共 10 小题（预计完成时间 5 分钟）。正确回答所有问题后截图给我们，稍后我们会在两个工作日内开通账号。

https://www.bilibili.com/video/BV1oK4y1x7AJ
（高性能计算平台新手上路指南，请在浏览器或 B 站客户端里打开）

可参阅的资料：

a) 用户文档：https://docs.hpc.sjtu.edu.cn

b) 学习视频：https://space.bilibili.com/483478550

c) HPC 网站：https://hpc.sjtu.edu.cn

d) 简短版使用手册（Cheat Sheet）：https://hpc.sjtu.edu.cn/Item/docs/Pi_GetStarted.pdf


提交 Hello world 单节点作业
===================================

以单节点的 OpenMP Hello world 为例，演示 π 集群作业提交过程。

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



登录可视化计算平台
==================

HPC Studio 可视化平台，集成 web shell、文件管理、作业提交、可视化应用等一站式服务。

登陆方法：

在浏览器中打开：\ `HPC Studio 可视化平台 <https://studio.hpc.sjtu.edu.cn>`__\  

详情请见：\ `HPC Studio 可视化平台使用方法 <../studio/basic.html>`__\ 


















