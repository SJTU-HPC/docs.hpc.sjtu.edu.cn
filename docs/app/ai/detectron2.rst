Detectron2
=======================

简介
----

Detectron2 是 Facebook AI Research 开源的计算机视觉库，它是基于 PyTorch 框架构建的。Detectron2 提供了一系列丰富的功能和灵活的组件，用于实现图像和视频中的目标检测、实例分割、关键点检测等任务。

Detectron2 构建在 Detectron 的基础上，并进行了全面的重写和重构，以提供更高的性能、更好的模块化设计和更强的扩展性。它采用了现代化的模型架构，如 Faster R-CNN、Mask R-CNN、Panoptic FPN 等，并提供了训练和推理的各种工具和接口。

这个文档将介绍如何在思源一号上安装GPU版Detectron2，并在A100加速卡上运行Inference Demo。


安装Detectron2
----------------

我们将申请思源一号上的1个计算节点用于执行安装流程。
Detectron2将被安装到名为 ```detectron2``` 的Conda环境中。

申请计算节点：

.. code:: bash

    $ srun -p 64c512g -n 1 --pty /bin/bash

在计算节点上加载模块，创建并激活 ```detectron2``` 环境：

.. code:: bash

    $ module load miniconda3
    $ conda create -n detectron2
    $ source activate detectron2

安装 PyTorch 2.0 GPU 版：

.. code:: bash

    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

安装与PyTorch所用CUDA版本匹配的detectron2依赖包：

.. code:: bash

    $ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    $ pip install opencv-python


安装detectron2：

.. code:: bash

    $ git clone https://github.com/facebookresearch/detectron2.git
    $ python -m pip install -e detectron2


以交互式方式使用detectron2:
----------------------------

我们将申请A100计算资源，激活 ```detectron2``` 环境后，运行detectron2 demo示例程序。

申请A100计算资源：

.. code:: bash

    $ srun -p a100 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=16 --pty /bin/bash

加载cuda模块，激活 ```detectron2``` 环境：

.. code:: bash

    $ module load miniconda3
    $ source activate detectron2
    $ module load cuda/11.8.0

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

测试demo算例，进入demo路径内，并准备一张测试输入图片上传至`./testimages/test1.jpg`内，并创建output目录,执行demo算例的命令如下：

.. code:: bash

    $ cd detectron2/demo
    $ python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml   --input ./testimages/test1.jpg --output ./output/  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

运行名为 ```demo.py`` 的算例，该算例耗时7秒钟，output目录内即可生成输出图片。

.. code:: bash

    $ python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml   --input ./testimages/test1.jpg --output ./output/  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
    ...
    [08/20 11:29:40 detectron2]: ./testimages/test1.jpg: detected 21 instances in 6.80s 100%|██████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.17s/it]
    $ ls output/
        test1.jpg
    

以SLURM批处理方式使用Detectron2
--------------------------------

我们将交互式运行Detectron2算例的过程整理成如下SLURM作业脚本，然后运行 ```sbatch Detectron2.slurm``` 提交：

.. code:: bash

    #!/bin/bash

    #SBATCH --job-name=Detectron2
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
    source activate detectron2
    module load cuda/11.8.0

    cd detectron2/demo
    python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml   --input ./testimages/test1.jpg --output ./output/  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
    

参考资料
--------

* Detectron2 https://detectron2.readthedocs.io/en/latest/tutorials/install.html
* PyTorch https://pytorch.org
