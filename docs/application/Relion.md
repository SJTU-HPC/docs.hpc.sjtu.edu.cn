# <center>Relion</center> 

-----

Relion是由MRC的Scheres在2012年发布的针对单颗粒冷冻电镜图片进行处理的框架。

## 使用Relion容器镜像

集群中已经预置了进行了编译优化的容器镜像，通过调用该镜像即可运行Relion作业，无需单独安装，目前版本为`relion-3.0.8`。该容器文件位于`/lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg`

### 使用singularity容器提交PyTorch作业

以下为在DGX-2上使用Relion的容器作业脚本示例，其中作业使用单节点并分配2块GPU：

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

IMAGE_PATH=/lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg

singularity run --nv $IMAGE_PATH relion_refine_mpi --version
```

我们假设这个脚本文件名为`pytorch_singularity.slurm`,使用以下指令提交作业。

```bash
$ sbatch pytorch_singularity.slurm
```

### 使用HPC Studio启动可视化界面

首先参照[可视化平台](../../login/HpcStudio/)开启远程桌面，并在远程桌面中启动终端，并输入以下指令：

```bash
singularity exec /lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg relion
```

## 参考文献

- [Relion](http://www2.mrc-lmb.cam.ac.uk/relion)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)