PyTorch Geometric (PyG)
=======================

简介
----

PyTorch Geometric (PyG) 是基于PyTorch的图神经网络(Graph Neural Networks, GNNs)训练工具包。
PyG依赖的PyTorch提供CPU和GPU版本，这个文档将介绍如何在思源一号上安装GPU版PyG 2.3.1，并在A100加速卡上运行算例。
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
    $ source activate pyg-gpu-a100

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

我们将申请A100计算资源，激活 ```pyg-gpu-a100``` 环境后，运行 PyG 示例程序。

申请A100计算资源：

.. code:: bash

    $ srun -p a100 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=16 --pty /bin/bash

加载模块，激活 ```pyg-gpu-a100``` 环境：

.. code:: bash

    $ module load miniconda3
    $ source activate pyg-gpu-a100

确认PyTorch版本高于1.12、PyTorch使用的CUDA版本与计算节点的GPU驱动版本相匹配：

.. code:: bash

    $ python -c "import torch; print(torch.__version__)"
    2.0.1
    $ python -c "import torch; print(torch.version.cuda)"
    11.8
    $ nvidia-smi | head -4
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+

下载并解压算例：

.. code:: bash

    $ wget https://codeload.github.com/pyg-team/pytorch_geometric/tar.gz/refs/tags/2.3.1 -O pyg-2.3.1.tar.gz
    $ tar xzvpf pyg-2.3.1.tar.gz
    $ cd pytorch_geometric-2.3.1/examples

运行名为 ```dna`` 的算例，该算例做运行200个Epoch训练，耗时约1分钟。

.. code:: bash

    $ python dna.py
    ...
    Epoch: 200, Train: 0.9945, Val: 0.8856, Test: 0.8584

以SLURM批处理方式使用PyG
------------------------

我们将交互式运行PyG算例的过程整理成如下SLURM作业脚本，然后运行 ```sbatch pyg.slurm``` 提交：

.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=pyg
    #SBATCH --partition=a100
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=16
    #SBATCH --gres=gpu:1
    #SBATCH --mail-type=end
    #SBATCH --mail-user=YOU@EMAIL.COM
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module load miniconda3
    source activate pyg-gpu-a100

    python dna.py

参考资料
--------

* PyTorch Geometric (PyG) https://pytorch-geometric.readthedocs.io/
* PyTorch https://pytorch.org
