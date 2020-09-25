# <center>gnuplot</center> 

-----

## 简介

Gnuplot is a command-driven interactive function plotting program. It can be used to plot functions and data points in both two- and three-dimensional plots in many different formats. It was originally created to allow scientists and students to visualize mathematical functions and data interactively, but has grown to support many non-interactive uses such as web scripting. It is also used as a plotting engine by third-party applications like Octave.


## Pi 上的 gnuplot

gnuplot 需要在 HPC Studio 可视化平台上使用。Pi 登陆节点不支持 gnuplot 显示。

HPC Studio 可视化平台通过浏览器访问：https://studio.hpc.sjtu.edu.cn

浏览器需为 chrome, firefox 或 edge。


## 使用 gnuplot

### 在 HPC Studio 上连接远程桌面

查看 Pi 上已编译的 GPU 版软件:
```bash
$ module avail relion
```

调用该模块:
```bash
$ module load relion/3.0.8-gcc-8.3.0-openmpi
```

### GPU Relion 的 Slurm 脚本

在 dgx2 队列上使用 1 块 gpu，并配比 6 cpu 核心

```bash
#!/bin/bash

#SBATCH -J relion
#SBATCH -p dgx2
#SBATCH -n 6 # number of tasks
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load cuda/9.0.176-gcc-4.8.5
module load openmpi/3.1.5-gcc-9.2.0
module load relion/3.0.8-gcc-8.3.0-openmpi

srun --mpi=pmi2 relion_refine_mpi (relion 的命令...)
```

###  GPU Relion 提交作业
```bash
$ sbatch slurm.test
```

## 使用 Relion 容器镜像

集群中已预置了编译优化的容器镜像，通过调用该镜像即可运行 Relion，无需单独安装，目前版本为 `relion-3.0.8`。该容器文件位于 `/lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg`

### 使用 singularity 容器提交 Relion 作业

示例：在 DGX-2 上使用 Relion 容器，作业使用单节点并分配 2 块 GPU：

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH -o %j.out
#SBATCH -e %j.err

IMAGE_PATH=/lustre/share/img/relion-3.0.8-cuda9.2-openmpi4.0.simg

singularity run --nv $IMAGE_PATH relion_refine_mpi --version
```

假设这个脚本文件名为 `relion_singularity.slurm`，使用以下指令提交作业

```bash
$ sbatch relion_singularity.slurm
```

### 使用 HPC Studio 启动可视化界面

参照[可视化平台](../../login/HpcStudio/)，登陆 HPC Studio，在顶栏选择 Relion：

![avater](../img/relion2.png)
![avater](../img/relion1.png)


## 参考链接

- [gnuplot 官网](http://www.gnuplot.info/)
- [Singularity 文档](https://sylabs.io/guides/3.5/user-guide/)
