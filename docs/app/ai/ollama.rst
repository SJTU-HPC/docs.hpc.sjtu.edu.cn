.. _ollama:

ollama
========

简介
----

ollama是一个开源的大型语言模型服务工具，它允许用户在自己的硬件环境中轻松部署和使用大规模预训练模型。Ollama 的主要功能是在Docker容器内部署和管理大型语言模型（LLM），使得用户能够快速地在本地运行这些模型。它简化了部署过程，通过简单的安装指令，用户可以执行一条命令就在本地运行开源大型语言模型，例如Llama 3。

- 能直接运行大模型，与大模型进行对话
- ollama 命令具有管理大模型的能力
- 本地大模型安全可靠
- 终端直接开始聊天
- 社区提供了支持 web api 方式访问 WebUI


在交我算上安装ollama
--------------------------

下载ollama-linux-amd64文件
^^^^^^^^^^^^^^^^^^^^^^^^^^

下载方法1：通过github链接下载

点击https://ollama.com/download/ollama-linux-amd64 即可下载。

下载方法2：交大云盘

我们已将此文件上传交大云盘，可以直接在交大云盘下载。
https://pan.sjtu.edu.cn/web/share/15be420e80487d8d972045042a32c90d

建议修改ollama-linux-amd64文件名为ollama，方便后续操作。可先在本地修改文件名后再上传，或上传超算后使用命令行修改：
mv ollama-linux-amd64 ollama

修改ollama文件可执行权限
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   $ chmod +x /path/to/ollama

备注：需要将path部分修改为存储ollama的路径

配置Path环境变量
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   $ export PATH=/path/to/ollama:$PATH

为方便使用，可以将该命令添加到.bashrc文件中

部署Llama3
---------------------------------

申请计算资源
^^^^^^^^^^^^^^^^^

llama3:8b推理需要1块 GPU卡,申请交互式计算资源的命令如下：

Pi 2.0： 

.. code:: bash

   $ srun -p dgx2 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=6 --pty /bin/bash

思源1号：

.. code:: bash

   $ srun -p a100 -N 1 -n 1 --gres=gpu:1 --cpus-per-task 16 --pty /bin/bash

运行ollama服务
^^^^^^^^^^^^^^^^^

.. code:: bash

   $ ollama serve

打开一个新的终端，进入计算节点并运行模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   $ ollama run llama3:8b

备注：初次运行 ollama run llama3:8b，需要下载一个 4.7G的文件，需要等待一定时间。

接下来就可以愉快地与大模型对话了！

.. image:: ../../img/hello_to_llm.png
   :alt: Llama3对话界面

参考资料
--------

-  `ollama 官网 <https://ollama.com/>`__
