# <center>LAMMPS</center> 

-----

## 简介

LAMMPS is a large scale classical molecular dynamics code, and stands for Large-scale Atomic/Molecular Massively Parallel Simulator. LAMMPS has potentials for soft materials (biomolecules, polymers), solid-state materials (metals, semiconductors) and coarse-grained or mesoscopic systems. It can be used to model atoms or, more generically, as a parallel particle simulator at the atomic, meso, or continuum scale.

## Pi 上的 LAMMPS

Pi 上有多种版本的 LAMMPS:

- ![cpu](https://img.shields.io/badge/-cpu-blue)  [cpu](#cpu-lammps)

- ![gpu](https://img.shields.io/badge/-gpu-green) [gpu](#gpu-lammps)

- ![arm](https://img.shields.io/badge/-arm-yellow) [arm](#arm-lammps)

## CPU 版本 LAMMPS

### ![cpu](https://img.shields.io/badge/-cpu-blue) CPU 版本

查看 Pi 上已编译的软件模块:
```bash
$ module avail lammps
```

调用该模块:
```bash
$ module load lammps/20200505-intel-19.0.4-impi
```

### ![cpu](https://img.shields.io/badge/-cpu-blue)  CPU 版本 Slurm 脚本
在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点：
```bash
#!/bin/bash

#SBATCH -J lammps_test
#SBATCH -p cpu
#SBATCH -n 80
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load intel-parallel-studio/cluster.2019.4-intel-19.0.4
module load lammps/20200505-intel-19.0.4-impi

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun lmp -i YOUR_INPUT_FILE
```

用下方语句提交作业
```bash
$ sbatch slurm.test
```

### ![cpu](https://img.shields.io/badge/-cpu-blue) （进阶）CPU 版本自行编译

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的 Lammps.

1. 从官网下载 lammps，推荐安装最新的稳定版：
```bash
$ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz
```

2. 由于登陆节点禁止运行作业和并行编译，请申请计算节点资源用来编译 lammps，并在编译结束后退出：
```bash
$ srun -p small -n 4 --pty /bin/bash
```

3. 加载 Intel-mpi 模块：
```bash
$ module purge
$ module load intel-parallel-studio/cluster.2019.4-intel-19.0.4
```

4. 编译 (以额外安装 USER-MEAMC 包为例)
```bash
$ tar xvf lammps-stable.tar.gz
$ cd lammps-XXXXXX
$ cd src
$ make					         #查看编译选项
$ make package                   #查看包
$ make yes-user-meamc            #"make yes-"后面接需要安装的 package 名字
$ make -j 4 intel_cpu_intelmpi   #开始编译
```

5. 测试脚本

编译成功后，将在 src 文件夹下生成 lmp_intel_cpu_intelmpi. 后续调用，请给该文件的路径，比如 `~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi`
```bash
#!/bin/bash

#SBATCH -J lammps_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load intel-parallel-studio/cluster.2019.4-intel-19.0.4

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i YOUR_INPUT_FILE
```


## ![gpu](https://img.shields.io/badge/-gpu-green) GPU 版本 LAMMPS

### GPU Singularity 版本（速度跟 intel CPU 版本基本相同）

Singularity 版本的 LAMMPS 针对 Tesla V100 的 GPU 做过优化，性能很好，LJ 和 EAM 的 Benchmark 与同等计算价格的 CPU 基本一样。建议感兴趣的用户可以针对自己的算例，测试 CPU 和 GPU 计算效率，然后决定使用哪一种平台。

Pi 集群已预置 NVIDIA GPU CLOUD 提供的优化镜像，调用该镜像即可运行 LAMMPS，无需单独安装，目前版本为 15Jun2020。该容器文件位于 /lustre/share/img/hpc/lammps_15Jun2020.sif

以下 slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu 核心，调用 Singularity 容器中的 LAMMPS：

```bash
#!/bin/bash

#SBATCH --job-name=lmp_test
#SBATCH --partition=dgx2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:2

ulimit -s unlimited
ulimit -l unlimited

IMAGE_PATH=/lustre/share/img/hpc/lammps_15Jun2020.sif

srun --mpi=pmi2 singularity run --nv $IMAGE_PATH \
lmp -k on g 2 t 12  -sf kk -pk kokkos comm device -in in.eam
```

其中，g 2 t 12 意思是使用 2 张 GPU 和 12 个线程。-sf kk -pk kokkos comm device 是 LAMMPS 的 kokkos 设置，可以用这些默认值

使用如下指令提交：

```bash
$ sbatch lammps_gpu.slurm
```

### ![gpu](https://img.shields.io/badge/-gpu-green)（进阶）GPU 版本自行编译

lammps GPU 支持很好，Pi 集群有先进的 dgx2 队列。感兴趣的用户可自行编译 GPU 版本 Lammps.

1. 从官网下载 lammps，推荐安装最新的稳定版：
```bash
$ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz
```

2. 由于登陆节点禁止运行作业和并行编译，请申请计算节点资源用来编译 lammps，并在编译结束后退出。GPU 编译需要在 dgx2 节点上：
```bash
$ srun -p dgx2 --gres=gpu:1 --ntasks-per-node=12 --pty /bin/bash
```

3. 加载 cuda, openmpi 和 gcc 模块：
```bash
$ module load openmpi/3.1.5-gcc-8.3.0
$ module load cuda/10.2.89-gcc-8.3.0         
$ module load gcc/8.3.0-gcc-4.8.5
```

4. 编译 (以额外安装 NEB 和 MEAMC 包为例)
```bash
$ tar xvf lammps-stable.tar.gz
$ cd lammps-XXXXXX
$ cd lib/gpu                     #

Makefile 第25行 CUDA_ARCH = -arch=sm_70
make -j12  -f Makefile.linux

cd ../../src
make yes-gpu 
make yes-replica
make yes-user-meamc
make yes-manybody

make -j12  -f Makefile.gpu
```

5. 测试脚本

编译成功后，将在 src 文件夹下生成 lmp_gpu. 后续调用，请给该文件的路径，比如 `~/lammps-3Mar20/src/lmp_gpu`
```bash
#!/bin/bash

#SBATCH --job-name=lmp_test
#SBATCH --partition=dgx2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:1

ulimit -s unlimited
ulimit -l unlimited

LMP=~/lammps-3Mar20/src/lmp_gpu

srun --mpi=pmi2 $LMP -sf gpu -pk gpu 1 -in in.eam
```

然后使用如下指令提交：

```bash
$ sbatch lammps_gpu.slurm
```






## 参考链接
- [LAMMPS 官网](https://lammps.sandia.gov/)
- [NVIDIA GPU CLOUD](ngc.nvidia.com)
- [Singularity 文档](https://sylabs.io/guides/3.5/user-guide/)


