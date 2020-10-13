# <center>Relion</center> 

-----

## 简介

Relion 是由 MRC 的 Scheres 在 2012 年发布的针对单颗粒冷冻电镜图片进行处理的框架。

## Pi 上的 Relion

查看 Pi 上已编译的 GPU 版软件:
```bash
$ module av relion
```

调用该模块:
```bash
$ module load relion/3.0.8
```

## 使用 GPU 版本的 Relion

在 dgx2 队列上使用 1 块 gpu，并配比 6 cpu 核心。脚本名称可设为 slurm.test

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge
module load relion/3.0.8

srun --mpi=pmi2 relion_refine_mpi (relion 的命令...)
```

使用以下指令提交作业

```bash
$ sbatch slurm.test
```


### 使用 HPC Studio 启动可视化界面

参照[可视化平台](../../login/HpcStudio/)，登陆 HPC Studio，在顶栏选择 Relion：

![avater](../img/relion2.png)
![avater](../img/relion1.png)


## 参考链接

- [Relion 官网](http://www2.mrc-lmb.cam.ac.uk/relion)
- [Singularity 文档](https://sylabs.io/guides/3.5/user-guide/)
