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
| 4.7.12.1  | |cpu|   | Spack    | miniconda2/4.7.12.1                   |
+-----------+---------+----------+---------------------------------------+
| 4.6.14    | |cpu|   | Spack    | miniconda2/4.6.14                     |
+-----------+---------+----------+---------------------------------------+
| 4.8.2     | |cpu|   | Spack    | `miniconda3/4.8.2`_                   |
+-----------+---------+----------+---------------------------------------+
| 4.7.12.1  | |cpu|   | Spack    | miniconda3/4.7.12.1                   |
+-----------+---------+----------+---------------------------------------+
| 4.6.14    | |cpu|   | Spack    | miniconda3/4.6.14                     |
+-----------+---------+----------+---------------------------------------+

运行示例
--------

.. _miniconda3/4.10.3:

思源一号集群 Conda
^^^^^^^^^^^^^^^^^^

在思源一号集群上使用如下命令:

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   which conda

.. _conda4aarch64/1.0.0-gcc-4.8.5:

ARM 集群 Conda
^^^^^^^^^^^^^^^

在 ARM 节点上使用如下命令：

.. code-block:: bash

   srun -p arm128c256g -n 4 --pty /bin/bash
   module load conda4aarch64/1.0.0-gcc-4.8.5
   which conda

.. _miniconda3/4.8.2:

π 集群 Conda
^^^^^^^^^^^^^

在 π 集群上使用如下命令:    

.. code-block:: bash

   srun -p small -n 4 --pty /bin/bash
   module load miniconda3/4.8.2
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

迁移Conda环境到思源一号
------------------------

迁移Conda环境需要导出环境到文件中，用 ``conda env create`` 从配置文件中来创建同样的环境。以从 π 集群迁移Conda环境到思源一号为例，需要用到 π 集群（旧环境）、sydata节点（数据互通）、思源一号（新环境）。

.. code-block:: bash

   $ π 集群
   conda env list                   # 查看当前存在哪些虚拟环境
   source activate pymol            # 激活用户环境
   conda list                       # 查看环境的包和软件
   conda env export > pymol.yaml    # 导出环境到配置文件
   $ sydata 节点                    # 数据互通
   scp user@data.hpc.sjtu.edu.cn:~/pymol.yaml ~/pymol.yaml
   $ 思源一号
   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3    # 加载Conda
   conda env list                   # 查看当前存在哪些虚拟环境
   conda env create -f pymol.yaml   # 从配置文件创建环境

参考教学视频
`思源一号Conda环境迁移 <https://vshare.sjtu.edu.cn/play/6761cf62659cba810143d4621f5026db>`_

Conda创建Python环境
------------------------

Conda可以方便的创建特定版本的环境，以Python为例。首先申请交互的计算资源，再使用Conda创建Python为3.7的环境，如下：

.. code-block:: bash

   srun -p 64c512g -n 4 --pty /bin/bash
   module load miniconda3/4.10.3
   conda create --name mypython python==3.7

新建一个 ``hello_python.slurm`` 的文件，内容为：

.. code-block:: bash

   #!/bin/bash
   #SBATCH -J hello-python
   #SBATCH -p 64c512g
   #SBATCH -o %j.out
   #SBATCH -e %j.err
   #SBATCH -n 1

   module load miniconda3/4.10.3
   source ~/.bashrc  # 如果显示的路径不是预期的，需要这一行
   source activate mypython
   which python
   python -c "print('hello world')"

``sbatch hello_python.slurm`` 提交作业，即可用新建的Python环境输出结果了。

通过pip安装Python扩展包
------------------------

以安装 ``PyMuPDF`` 为例，如果 ``Conda`` 中找不到相关的 ``Python`` 包或者没有需要的版本，可以用 ``pip`` 安装。

.. code-block:: bash

   source activate env4test         # 激活虚拟环境env4test
   conda search pymupdf             # 找不到相关的包
   conda search -c tc06580 pymupdf  # 指定源搜索，只有1.17.0版本的
   which pip                        # 确定有安装pip，一般conda创建的Python环境都会有pip的
   pip install pymupdf              # 使用pip安装Python扩展包
   pip install -r requirements.txt  # 使用pip批量安装requirements.txt中的软件包
   pip list | grep -i pymupdf       # 安装成功，当前为1.19.4版本

.. tip:: 
   
   建议特定的一个或几个软件创建一个单独的环境，方便管理与使用。
   
   可以到Anaconda页面搜索是否有对应软件的源 https://anaconda.org/search


常见问题
----------------

1. 在一个conda环境中同时安装软件A与软件B，存在conflict问题
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A:** 建议新建环境分开进行安装。

2. 软件运行提示缺少 xxx.so 库
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A:** conda search 查找不同的版本及build信息，conda install 指定版本及build进行安装测试。

3. 安装的软件版本不支持GPU或者不支持python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A:** conda list 查看已安装的版本及build信息，确认build是GPU或python；conda install 指定版本、源及build进行安装测试。


   
参考资料
--------

-  `Conda 文档 <https://conda.io/en/latest/index.html>`__
