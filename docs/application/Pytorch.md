# PyTorch

PyTorch 是一个 Python 优先的深度学习框架，也是使用 GPU 和 CPU 优化的深度学习张量库，能够在强大的 GPU 加速基础上实现张量和动态神经网络。同时，PyTorch 主要为开发者提供两种高层面的功能：

1. 使用强大的 GPU 加速的 Tensor 计算（类似 numpy）；
2. 构建 autograd 系统的深度神经网络。

通常，人们使用 PyTorch 的原因通常有二：

1. 作为 numpy 的替代，以便使用强大的 GPU；
2. 将其作为一个能提供最大的灵活性和速度的深度学习研究平台。

## 作业脚本示例

以下为在DGX-2上使用PyTorch的作业脚本pytorch.slurm示例，其中作业使用单节点并分配2块GPU：

```
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

其中`/lustre/share/img/pytorch-19.10-py3.simg`是[NVIDIA GPU CLOUD](https://ngc.nvidia.com/)提供的优化镜像，目前版本为`pytorch-1.3.0`。

并使用如下指令提交：

```
$ sbatch pytorch.slurm
```

## 参考文献

- [pytorch官方网站](https://pytorch.org/)
- [NVIDIA GPU CLOUD](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)