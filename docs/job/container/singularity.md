# <center>在集群上使用Singularity<center/>

-------

## 高性能容器Singularity 

Singularity是劳伦斯伯克利国家实验室专门为大规模、跨节点HPC和DL工作负载而开发的容器化技术。具备轻量级、快速部署、方便迁移等诸多优势，且支持从Docker镜像格式转换为Singularity镜像格式。与Docker的不同之处在于：

1.  Singularity同时支持root用户和非root用户启动，且容器启动前后，用户上下文保持不变，这使得用户权限在容器内部和外部都是相同的。
2.  Singularity强调容器服务的便捷性、可移植性和可扩展性，而弱化了容器进程的高度隔离性，因此量级更轻，内核namespace更少，性能损失更小。

本文将向大家介绍在集群上使用Singularity的方法。

如果我们可以提供任何帮助，请随时联系[hpc邮箱](hpc@sjtu.edu.cn)。

## 镜像准备

首先我们需要准备Singularity镜像。如果镜像来自于[Docker Hub](https://hub.docker.com/)，则可以直接在集群中使用如下命令制作镜像。

```
$ singularity build ubuntu.simg docker://ubuntu
INFO:    Starting build...
Getting image source signatures
...
INFO:    Creating SIF file...
INFO:    Build complete: ubuntu.simg
```

如果需要自行构建镜像或者修改现有镜像，因为其过程需要root权限，我们建议使用个人的Linux环境进行镜像构建然后传至集群。在完成镜像构建后，再将镜像上传至集群。

我们在集群中预置了以下软件的Singularity的镜像。

| 软件  | 位置  |
| ---- | ---- |
| PyTorch | /lustre/share/img/pytorch-19.10-py3.simg |
| Gromacs | /lustre/share/img/gromacs-2018.2.simg  |
| vmd | /lustre/share/img/vmd-1.9.3.simg |
| octave | /lustre/share/img/octave-4.2.2.simg | 
|openfoam| /lustre/share/img/openfoam-6.simg |

## 任务提交

```
#!/bin/bash

#SBATCH --job-name=test_singularity
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -n 1
#SBATCH --exclusive

ulimit -l unlimited
ulimit -s unlimited

singularity run /lustre/share/img/pytorch-19.10-py3.simg python -c "import torch;print(torch.__version__)"
```

## 参考文献
 - [Singularity Quick Start](https://sylabs.io/guides/3.4/user-guide/quick_start.html)
 - [Docker Hub](https://hub.docker.com/)
