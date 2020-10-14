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
module av lammps
```

推荐使用 lammps/2020-cpu，经测试，该版本在 Pi 2.0 上运行速度最好，且安装有丰富的 LAMMPS package：

ASPHERE BODY CLASS2 COLLOID COMPRESS CORESHELL DIPOLE
        GRANULAR KSPACE MANYBODY MC MISC MLIAP MOLECULE OPT PERI
        POEMS PYTHON QEQ REPLICA RIGID SHOCK SNAP SPIN SRD VORONOI
        USER-BOCS USER-CGDNA USER-CGSDK USER-COLVARS USER-DIFFRACTION
        USER-DPD USER-DRUDE USER-EFF USER-FEP USER-MEAMC USER-MESODPD
        USER-MISC USER-MOFFF USER-OMP USER-PHONON USER-REACTION
        USER-REAXC USER-SDPD USER-SPH USER-SMD USER-UEF USER-YAFF

调用该模块:
```bash
module load lammps/2020-cpu
```

### ![cpu](https://img.shields.io/badge/-cpu-blue)  CPU 版本 Slurm 脚本
在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点。脚本名称可设为 slurm.test
```bash
#!/bin/bash
#SBATCH --job-name=lmp_test
#SBATCH --partition=cpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=40

module purge
module load lammps/2020-cpu

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 lmp -i YOUR_INPUT_FILE
```

用下方语句提交作业
```bash
sbatch slurm.test
```

### ![cpu](https://img.shields.io/badge/-cpu-blue) （进阶）CPU 版本自行编译

若对 lammps 版本有要求，或需要特定的 package，可自行编译 Intel 版本的 Lammps.

1. 从官网下载 lammps，推荐安装最新的稳定版：
```bash
$ wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz
```

2. 由于登陆节点禁止运行作业和并行编译，请申请计算节点资源用来编译 lammps，并在编译结束后退出：
```bash
$ srun -p small -n 8 --pty /bin/bash
```

3. 加载 Intel 模块：
```bash
$ module purge
$ module load intel-parallel-studio/cluster.2019.4-intel-19.0.4
```

4. 编译 (以额外安装 MANYBODY 和 USER-MEAMC 包为例)
```bash
$ tar xvf lammps-stable.tar.gz
$ cd lammps-XXXXXX
$ cd src
$ make					         #查看编译选项
$ make package                   #查看包
$ make yes-user-meamc            #"make yes-"后面接需要安装的 package 名字
$ make yes-manybody
$ make ps                        #查看计划安装的包列表 
$ make -j 8 intel_cpu_intelmpi   #开始编译
```

5. 测试脚本

编译成功后，将在 src 文件夹下生成 lmp_intel_cpu_intelmpi. 后续调用，请给该文件的路径，比如 `~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi`。脚本名称可设为 slurm.test
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

### GPU 版本速度跟 intel CPU 版本基本相同

Pi 上提供了 GPU 版本的 LAMMPS 2020。经测试，LJ 和 EAM 两 Benchmark 算例与同等计算费用的 CPU 基本一样。建议感兴趣的用户针对自己的算例，测试 CPU 和 GPU 计算效率，然后决定使用哪一种平台。

以下 slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu 核心，使用 GPU 版 LAMMPS。脚本名称可设为 slurm.test

```bash
#!/bin/bash

#SBATCH --job-name=lmp_test
#SBATCH --partition=dgx2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

ulimit -s unlimited
ulimit -l unlimited

module load lammps/2020-dgx

srun --mpi=pmi2 lmp -in in.eam
```

使用如下指令提交：

```bash
$ sbatch slurm.test
```

## ![gpu](https://img.shields.io/badge/-gpu-green) GPU 版本 LAMMPS + kokkos

### GPU 版本速度跟 intel CPU 版本基本相同

Pi 上提供了 GPU + kokkos 版本的 LAMMPS 15Jun2020。采用容器技术，使用 LAMMPS 官方提供给 NVIDIA 的镜像，针对 Tesla V100 的 GPU 做过优化，性能很好。经测试，LJ 和 EAM 两 Benchmark 算例与同等计算费用的 CPU 基本一样。建议感兴趣的用户针对自己的算例，测试 CPU 和 GPU 计算效率，然后决定使用哪一种平台。

以下 slurm 脚本，在 dgx2 队列上使用 2 块 gpu，并配比 12 cpu 核心，使用 GPU kokkos 版的 LAMMPS。脚本名称可设为 slurm.test

```bash
#!/bin/bash

#SBATCH --job-name=lmp_test
#SBATCH --partition=dgx2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2

ulimit -s unlimited
ulimit -l unlimited

module load lammps/2020-dgx-kokkos

srun --mpi=pmi2 lmp -k on g 2 t 12  -sf kk -pk kokkos comm device -in in.eam
```

其中，g 2 t 12 意思是使用 2 张 GPU 和 12 个线程。-sf kk -pk kokkos comm device 是 LAMMPS 的 kokkos 设置，可以用这些默认值

使用如下指令提交：

```bash
$ sbatch slurm.test
```





## 参考链接
- [LAMMPS 官网](https://lammps.sandia.gov/)
- [NVIDIA GPU CLOUD](ngc.nvidia.com)



