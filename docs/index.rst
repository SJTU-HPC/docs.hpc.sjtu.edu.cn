======================================
上海交大交我算HPC+AI平台用户文档
======================================

上海交通大学交我算平台是校级计算公共服务平台，成立于 2013 年，旨在对校内大规模科学与工程计算需求提供技术支撑。

欢迎致谢交我算平台
======================================

致谢模板及优秀论文见平台成果展示：`https://hpc.sjtu.edu.cn/Item/Science.htm <https://hpc.sjtu.edu.cn/Item/Science.htm>`_ 

交我算HPC+AI平台
===================

- “思源一号”高性能计算平台。2022年新上线的“思源一号”集群总算力 6 PFLOPS（每秒千万亿次），是目前国内高校第一的超算集群，TOP500 榜单排名第132位。CPU 采用双路 Intel Xeon ICX Platinum 8358 32 核，主频 2.6GHz，共938个计算节点；GPU采用NVIDIA HGX A100, 共92块GPU卡。计算节点之间使用Mellanox 100 Gbps Infiniband HDR 高速互联，并行存储的聚合存储能力达10 PB。

- π 2.0超算平台。 π 2.0 超算系统于 2019 年上线，双精度浮点数理论性能 2.1 PFLOPS，拥有 658 个双路节点和 1316 颗第二代英特尔至强金牌 6248 处理器，并配以英特尔 Omni-Path 架构的 100 Gbps 高速网络互连，以及全闪存的 NVMeLustre 存储系统，体现了强大的计算能力和先进的设计理念。

- AI 平台。AI 平台由 8 台 NVIDIA DGX-2 服务器组成，双精度计算能力达 1 PFLOPS，张量计算能力达 16 PFLOPS。每台 DGX-2 配置 16 块 Tesla V100 GPU 加速卡，2 颗 Intel 至强铂金 8168 CPU，1.5 TB DDR4 内存，30 TB NVMe SSD 和 512GB HBM2 显存。

- ARM 平台。ARM 平台于2021年上线，基于 ARM 处理器构建，是国内首台基于 ARM 处理器的校级超算。共 100 个计算节点，与 π 2.0和AI平台实现共享登录、共享 Lustre 文件系统和共享 Slurm 作业调度系统。ARM 超算单节点配备 128 核（2.6 GHz）、256 GB 内存（16 通道 DDR4-2933）、240 GB 本地硬盘。

文档使用
===================

`docs <https://docs.hpc.sjtu.edu.cn/>`_ 为HPC和AI平台使用文档，为用户提供快速上手指导和问题解答。可访问 `上海交通大学 HPC 站点 <https://hpc.sjtu.edu.cn/>`_ 获取HPC和AI平台更多信息。

1. :doc:`quickstart/index`
2. :doc:`system/index`
3. :doc:`accounts/index`
4. `密码 <accounts/index.html#id7>`_ 
5. :doc:`login/index`
6. :doc:`studio/rdp`
7. :doc:`job/index`
8. :doc:`app/index`
9. :doc:`app/compilers_and_languages/gnu`
10. :doc:`app/compilers_and_languages/intel`
11. :doc:`faq/index`

注意事项
===================

- 交我算HPC+AI平台禁止运行军工项目等涉密计算任务。
- 欢迎邮件联系我们：hpc[AT]sjtu.edu.cn

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart/index

.. toctree::
   :maxdepth: 2
   :hidden:

   system/index

.. toctree::
   :maxdepth: 2
   :hidden:

   accounts/index

.. toctree::
   :maxdepth: 2
   :hidden:

   studio/index   

.. toctree::
   :maxdepth: 2
   :hidden:

   login/index

.. toctree::
   :maxdepth: 2
   :hidden:

   transport/index

.. toctree::
   :maxdepth: 2
   :hidden:

   job/index

.. toctree::
   :maxdepth: 2
   :hidden:

   container/index

.. toctree::
   :maxdepth: 2
   :hidden:

   app/index

.. toctree::
   :maxdepth: 2
   :hidden:

   faq/index

.. toctree::
   :maxdepth: 2
   :hidden:

   download/index
