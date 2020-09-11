# <center>Octave</center>

---------

GNU Octave是一种采用高级编程语言的主要用于数值分析的软件。Octave有助于以数值方式解决线性和非线性问题，并使用与MATLAB兼容的语言进行其他数值实验。它也可以作为面向批处理的语言使用。因为它是GNU计划的一部分，所以它是GNU通用公共许可证条款下的自由软件。

## 使用singularity容器提交Octave作业

以下是基于Singularity的作业脚本`octave_singularity.slurm`示例：

```bash
#!/bin/bash
#SBATCH -J octave_test
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

IMAGE_PATH=/lustre/share/img/octave.simg

ulimit -s unlimited
ulimit -l unlimited

singularity run $IMAGE_PATH octave [FILE_NAME]
```

并使用如下指令提交：

```bash
$ sbatch octave_singularity.slurm
```


## 使用singularity容器提交Octave交互式作业

可以通过如下指令提交Octave交互式作业：

```bash
srun -p cpu -N 1 --exclusive --pty singularity run /lustre/share/img/octave.simg octave-cli
```

## 使用HPC Studio启动Octave可视化界面

首先参照[可视化平台](../../login/HpcStudio/)开启远程桌面，并在远程桌面中启动终端，并输入以下指令：

```bash
singularity run /lustre/share/img/octave.simg octave
```

## 参考文献

- [Octave官方网站](https://www.gnu.org/software/octave/)
- [Singularity文档](https://sylabs.io/guides/3.5/user-guide/)
