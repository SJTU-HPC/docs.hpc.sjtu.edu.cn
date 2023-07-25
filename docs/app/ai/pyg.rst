PyTorch Geometric (PyG)
=======================

简介
----

PyTorch Geometric (PyG) 是基于PyTorch的图神经网络(Graph Neural Networks, GNNs)训练工具包。
PyG依赖的PyTorch提供CPU和GPU版本，这个文档将介绍如何在思源一号上安装GPU版PyTorch，并在A100加速卡上运行算例。
如果您需要在CPU或者使用DGX2上的V100加速卡上运行PyG，您可以咨询技术服务团队。

安装PyG
--------

我们将申请思源一号上的1个计算节点用于执行安装流程。
PyG将被安装到名为 ```pyg-gpu-a100``` 的Conda环境中。

申请计算节点：

.. code:: bash

    $ srun -p 64c512g -n 1 --pty /bin/bash

在计算节点上加载模块，创建并激活 ```pyg-gpu-a100``` 环境：

.. code:: bash

    $ module load miniconda3
    $ conda create -n pyg-gpu-a100

安装 PyTorch 2.0 GPU 版：

.. code:: bash

    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

安装与PyTorch所用CUDA版本匹配的PyG依赖包：

.. code:: bash

    $ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

安装PyG：

.. code:: bash

    $ pip install torch_geometric

以交互式方式使用PyG
-------------------


以SLURM批处理方式使用PyG
------------------------

参考资料
--------

* PyTorch Geometric (PyG) https://pytorch-geometric.readthedocs.io/
