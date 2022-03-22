.. _Conda:

Conda
=====

简介
----

Conda是一个可在Linux、macOS和Windows上运行的开源软件包管理和环境管理系统。Conda可快速安装、运行和升级软件包及其依赖包。Conda可在本地计算机上轻松地进行创建、保存、加载和切换环境。它是为Python程序创建的，但是也可以打包和分发适用于任何语言的软件。

Conda作为软件包管理器，可以帮助用户查找和安装软件包。如果用户需要一个使用其他版本的Python的软件包，无需切换到其他环境管理器，因为Conda也是环境管理器，仅需几个命令，用户就可以设置一个完全独立的环境来运行该不同版本的Python，同时继续在正常环境中运行用户通常的Python版本。

可用的版本
----------

+-----------+---------+----------+---------------------------------------+
| 版本      | 平台    | 构建方式 | 模块名                                |
+===========+=========+==========+=======================================+
| 4.7.12.1  | |cpu|   | Spack    | miniconda2/4.7.12.1 思源一号          |
+-----------+---------+----------+---------------------------------------+
| 4.10.3    | |cpu|   | Spack    | `miniconda3/4.10.3`_ 思源一号         |
+-----------+---------+----------+---------------------------------------+
| 4.5.12    | |arm|   | Spack    | `conda4aarch64/1.0.0-gcc-4.8.5`_      |
+-----------+---------+----------+---------------------------------------+
| 4.7.12.1  | |cpu|   | Spack    | miniconda2/4.7.12.1-gcc-4.8.5         |
+-----------+---------+----------+---------------------------------------+
| 4.6.14    | |cpu|   | Spack    | miniconda2/4.6.14-gcc-4.8.5           |
+-----------+---------+----------+---------------------------------------+
| 4.8.2     | |cpu|   | Spack    | `miniconda3/4.8.2-gcc-4.8.5`_         |
+-----------+---------+----------+---------------------------------------+
| 4.7.12.1  | |cpu|   | Spack    | miniconda3/4.7.12.1-gcc-4.8.5         |
+-----------+---------+----------+---------------------------------------+
| 4.6.14    | |cpu|   | Spack    | miniconda3/4.6.14-gcc-4.8.5           |
+-----------+---------+----------+---------------------------------------+

运行示例
--------

.. _miniconda3/4.10.3:

思源一号集群 Conda
^^^^^^^^^^^^^^^^^^

在思源一号集群上使用如下命令:    

.. code-block:: bash

   module purge
   module load miniconda3/4.10.3
   which conda

.. _conda4aarch64/1.0.0-gcc-4.8.5:

ARM 集群 Conda
^^^^^^^^^^^^^^^

在 ARM 节点上使用如下命令：

.. code-block:: bash

   module purge
   module load conda4aarch64/1.0.0-gcc-4.8.5
   which conda

.. _miniconda3/4.8.2-gcc-4.8.5:

π 集群 Conda
^^^^^^^^^^^^^

在 π 集群上使用如下命令:    

.. code-block:: bash

   module purge
   module load miniconda3/4.8.2-gcc-4.8.5
   which conda

Conda常用命令
-------------

.. code-block:: bash

   conda list                       # 查看安装了哪些包
   conda env list                   # 查看当前存在哪些虚拟环境
   conda create -n env4test         # 创建一个名为env4test的虚拟环境
   source activate env4test         # 激活虚拟环境env4test
   conda deactivate                 # 退出虚拟环境
   conda search bwa -c bioconda     # 查找名为bwa的包，并指定bioconda源
   conda install bwa -c bioconda -n env4test # 指定从bioconda源中下载安装bwa，安装在env4test虚拟环境中
   conda remove -n env4test bwa     # 删除虚拟环境中的bwa包
   conda remove -n env4test --all   # 删除虚拟环境env4test(包括其中的所有的包)

.. tip:: 
   
   建议特定的一个或几个软件创建一个单独的环境，方便管理与使用。

参考资料
--------

-  `Conda 文档 <https://conda.io/en/latest/index.html>`__
