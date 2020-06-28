# <center>LAMMPS</center> 

-----

## 简介

LAMMPS is a large scale classical molecular dynamics code, and stands for Large-scale Atomic/Molecular Massively Parallel Simulator. LAMMPS has potentials for soft materials (biomolecules, polymers), solid-state materials (metals, semiconductors) and coarse-grained or mesoscopic systems. It can be used to model atoms or, more generically, as a parallel particle simulator at the atomic, meso, or continuum scale.

## Pi 上的 LAMMPS
查看 Pi 上已编译的软件模块:
```bash
$ module avail lammps
```

调用该模块:
```bash
$ module load lammps/20190807-intel-19.0.5-impi
```

## Pi 上的 Slurm 脚本 slurm.test
在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点：
```bash
#!/bin/bash

#SBATCH -J QE_test
#SBATCH -p cpu
#SBATCH -n 80
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
module load lammps/20190807-intel-19.0.5-impi

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun lmp -i YOUR_INPUT_FILE
```

## Pi 上提交作业
```bash
$ sbatch slurm.test
```

## 自行编译 Lammps

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
$ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
```

4. 编译 (以额外安装 USER-MEAMC 包为例)
```bash
$ tar xvf lammps-stable.tar.gz
$ cd lammps-XXXXXX
$ cd src
$ make					            #查看编译选项
$ make package                   #查看包
$ make yes-user-meamc            #"make yes-"后面接需要安装的 package 名字
$ make -j 4 intel_cpu_intelmpi   #开始编译
```
   编译成功后，将在 src 文件夹下生成 lmp_intel_cpu_intelmpi. 后续调用，请给该文件的路径，比如 ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi

5. 测试脚本
```bash
#!/bin/bash

#SBATCH -J lammps_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i YOUR_INPUT_FILE
```

## 参考链接
- [LAMMPS 官网](https://lammps.sandia.gov/)



