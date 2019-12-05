# <center>PyTorch</center>

-----------

PyTorch 是一个 Python 优先的深度学习框架，也是使用 GPU 和 CPU 优化的深度学习张量库，能够在强大的 GPU 加速基础上实现张量和动态神经网络。同时，PyTorch 主要为开发者提供两种高层面的功能：

1. 使用强大的 GPU 加速的 Tensor 计算（类似 numpy）；
2. 构建 autograd 系统的深度神经网络。

通常，人们使用 PyTorch 的原因通常有二：

1. 作为 numpy 的替代，以便使用强大的 GPU；
2. 将其作为一个能提供最大的灵活性和速度的深度学习研究平台。

## 使用miniconda安装PyTorch

创建名为`pytorch-env`的虚拟环境，激活虚拟环境，然后安装pytorch。

```bash
$ module load miniconda3
$ conda create -n pytorch-env
$ source activate pytorch-env
$ conda install pytorch torchvision -c pytorch
```

## 使用miniconda提交PyTorch作业

以下为在DGX-2上使用PyTorch的虚拟环境作业脚本示例，其中作业使用单节点并分配2块GPU：

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:2

module load miniconda3
source activate pytorch-env

python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'
```

我们假设这个脚本文件名为`pytorch_conda.slurm`,使用以下指令提交作业。

```bash
$ sbatch pytorch_conda.slurm
```

## 使用singularity容器中的PyTorch

集群中已经预置了[NVIDIA GPU CLOUD](https://ngc.nvidia.com/)提供的优化镜像，通过调用该镜像即可运行PyTorch作业，无需单独安装，目前版本为`pytorch-1.3.0`。该容器文件位于`/lustre/share/img/pytorch-19.10-py3.simg`


## 使用singularity容器提交PyTorch作业

以下为在DGX-2上使用PyTorch的容器作业脚本示例，其中作业使用单节点并分配2块GPU：

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:2

IMAGE_PATH=/lustre/share/img/pytorch-19.10-py3.simg

singularity run --nv $IMAGE_PATH python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'
```

我们假设这个脚本文件名为`pytorch_singularity.slurm`,使用以下指令提交作业。

```bash
$ sbatch pytorch_singularity.slurm
```

## 参考文献

- [PyTorch官方网站](https://pytorch.org/)
- [NVIDIA GPU CLOUD](ngc.nvidia.com)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)