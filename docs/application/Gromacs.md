# <center>GROMACS</center>

GROMACS是一种分子动力学应用程序，可以模拟具有数百至数百万个粒子的系统的牛顿运动方程。GROMACS旨在模拟具有许多复杂键合相互作用的生化分子，例如蛋白质，脂质和核酸。有关GROMACS的更多信息，请访问[http://www.gromacs.org/](http://www.gromacs.org/)。

## 模块加载方法

可以使用module直接加载GROMACS应用，默认版本为2019.2：

```shell
$ module purge
$ module load gromacs
```

## 作业脚本示例

单节点作业脚本示例gromacs.slurm如下：

```
#!/bin/bash
#SBATCH -J gromacs_cpu_test
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1

module purge
module load gromacs

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 gmx_mpi mdrun -deffnm test -ntomp 1
```

并使用如下指令提交：

```
$ sbatch gromacs.slurm
```

多节点运行，只需要更改`#SBATCH -N 1`中的`1`至你需要的节点数量。

## 使用GPU运行GROMACS

如需使用GPU运行GROMACS，需要指定使用dgx2分区。以下是基于Singularity的作业脚本gromacs.slurm示例：

```
#!/bin/bash
#SBATCH -J gromacs_gpu_test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:1

IMAGE_PATH=/lustre/share/img/gromacs-2018.2.simg

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 singularity run --nv $IMAGE_PATH gmx mdrun -deffnm test -ntomp 1
```

其中`/lustre/share/img/gromacs-2018.2.simg`是[NVIDIA GPU CLOUD](https://ngc.nvidia.com/)提供的优化镜像，目前版本为2018.2。

并使用如下指令提交：

```
$ sbatch gromacs.slurm
```

## 参考文献

- [gromacs官方网站](http://www.gromacs.org/)
- [NVIDIA GPU CLOUD](https://ngc.nvidia.com/catalog/containers/hpc:gromacs)