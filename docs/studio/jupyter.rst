Jupyter
================

Jupyter是一个非营利组织，旨在“为数十种编程语言的交互式计算开发开源软件，开放标准和服务”。2014年由Fernando
Pérez从IPython中衍生出来，Jupyter支持几十种语言的执行环境。

Jupyter
Project的名称是对Jupyter支持的三种核心编程语言的引用，这三种语言是Julia、Python和R，也是对伽利略记录发现木星的卫星的笔记本的致敬。Jupyter项目开发并支持交互式计算产品Jupyter
Notebook、JupyterHub和JupyterLab，这是Jupyter Notebook的下一代版本。

登录HPC
Studio平台后，可以在内置应用中选择\ ``Jupyter``\ 或\ ``Jupyer (GPU)``\ ，均支持\ ``Jupyter Notebook``\ 和\ ``JupyterLab``\ 。

在 Jupyter 中使用预置环境
-------------------------

已有三个预置环境，可供用户使用：

预置 PyTorch 环境
~~~~~~~~~~~~~~~~~

=========== ========
环境        版本
=========== ========
python      3.8.3
cudatoolkit 10.1.243
pytorch     1.5.0
torchvision 0.6.0
numpy       1.18.1
pandas      1.0.4
pillow      7.1.2
scipy       1.4.1
matplotlib  3.2.1
seaborn     0.10.1
=========== ========

预置 TensorFlow 环境
~~~~~~~~~~~~~~~~~~~~

=========== ========
环境        版本
=========== ========
python      3.8.3
cudatoolkit 10.1.243
cudnn       7.6.5
tensorflow  2.2.0
tensorboard 2.2.2
numpy       1.18.5
pandas      1.0.4
pillow      7.1.2
scipy       1.4.1
matplotlib  3.2.1
seaborn     0.10.1
=========== ========

预置 R 环境
~~~~~~~~~~~

==== =====
环境 版本
==== =====
R    3.6.1
==== =====

在 Jupyter 中使用自定义的环境
-----------------------------

新建环境（或使用已有环境）:

.. code:: shell

   $ module load miniconda3
   $ conda create -n test-env
   $ source activate test-env

安装并注册为\ ``jupter kernel``\ ：

.. code:: shell

   (test-env) $ conda install ipykernel
   (test-env) $ python -m ipykernel install --user --name test-env --display-name "Test Environment"

然后可以在\ ``Jupyter``\ 中选择名为\ ``Test Environment``\ 的Kernel进行计算。

如果环境需要依赖\ ``NVIDIA CUDA Toolkit``\ 或\ ``NVIDIA cuDNN``\ ，可以使用\ ``conda``\ 进行安装：

.. code:: shell

   (test-env) $ conda install cudatoolkit=10.1 cudnn

在 Jupyter 中使用自定义 R 环境
------------------------------

新建环境（或使用已有环境）:

.. code:: shell

   $ module load miniconda3
   $ conda create -n r-test-env
   $ source activate r-test-env
   $ (r-test-env) $ conda install -c r r-essentials

安装并注册为\ ``jupter kernel``\ ：

.. code:: shell

   (test-env) $ R
   > install.packages('IRkernel')
   > IRkernel::installspec(name = 'r-test-env', displayname = 'R 3.6.1')

然后可以在\ ``Jupyter``\ 中选择名为\ ``R 3.6.1``\ 的Kernel进行计算。

参考资料
--------

-  `Jupyter Wikepedia <https://zh.wikipedia.org/wiki/Jupyter>`__
-  `Jupyter Home <https://jupyter.org/>`__
