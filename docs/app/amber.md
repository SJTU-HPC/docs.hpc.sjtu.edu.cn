# <center>Amber</center>

------

## 简介

Amber (Assisted Model Building with Energy Refinement) is the collective name for a suite of programs designed to carry out molecular mechanical force field simulations, particularly on biomolecules. 


## Pi 上的 Amber
由于 Amber 是需要版权的软件，Pi 上不提供。需要用户自行获取版权并安装。安装方法见本文档后面部分。

## Pi 上的 Slurm 脚本 slurm.test
（版本：GNU + cpu）<br>
在 cpu 队列上，总共使用 80 核 (n = 80)<br>
cpu 队列每个节点配有 40 核，所以这里使用了 2 个节点：
```bash
#!/bin/bash

#SBATCH -J amber_test
#SBATCH -p cpu
#SBATCH -n 80
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load openmpi/3.1.5-gcc-9.2.0

ulimit -s unlimited
ulimit -l unlimited

source {YOUR amber.sh}
srun --mpi=pmi2 {pmemd.MPI ... YOUR AMBER COMMANDS}
```

（版本：GNU + gpu）<br>
在 dgx2 队列上，使用一张卡：
```bash
#!/bin/bash

#SBATCH -J amber_gpu_test
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 6 # number of tasks
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

module purge
module load cuda/9.0.176-gcc-4.8.5
module load openmpi/3.1.5-gcc-9.2.0

ulimit -s unlimited
ulimit -l unlimited

source {path/to/your/amber.sh}
srun --mpi=pmi2 {YOUR AMBER CUDA COMMANDS; eg: pmemd.cuda.MPI -ng 6 ... }
```

## Pi 上提交作业
```bash
$ sbatch slurm.test
```

## Amber 安装

安装前请移除 .bashrc 不必要的内容，包括 module load 与 export 等等

- 准备 amber18 源代码
```bash
$ tar xvf amber18.tar.gz
$ cd amber18
$ make veryclean
```
!!! tip
      `make veryclean`将移除大量编译过的临时文件等内容, 具体查看[Amber文档](http://ambermd.org/doc12/Amber18.pdf)
		 
- 编译 Amber 非常消耗计算资源，请登陆到 CPU 节点
```bash
$ srun -p cpu -N 1 --exclusive --pty /bin/bash
```

- 安装 Amber18 的串行版本 (不可跳过)
```bash
$ export AMBERHOME=$(pwd)  ## make sure you are in the amber18 directory extracted
$ ./configure --no-updates -noX11 gnu
$ source ./amber.sh
$ make -j 40 && make install   #change 40 to total ncore
```

!!! tip
      `--no-updates` 表示跳过 "download & install updates"。
      如果提示是否自动下载安装 miniconda, 请根据自己需求选择 YES or NO

!!! tip
      如果您的任务规模较小，仅需编译串行版本Amber，那么至此编译工作已经完成。但我们强烈建议您继续编译MPI或CUDA版本。

## 编译 MPI 版本

- 安装 Amber18 的 MPI 版本
```bash
$ module load openmpi
$ ./configure --no-updates -noX11 -mpi gnu
$ make -j 40 && make install
```

## 编译 CUDA 版本

- 安装 Amber18 的 CUDA 版本
```bash
$ module load cuda/9.0.176-gcc-4.8.5
$ export LIBRARY_PATH="/lustre/opt/cascadelake/linux-centos7-x86_64/gcc-4.8.5/cuda-9.0.176-pusmroeeme62xntggzjygame4htcbil7/lib64/stubs:${LIBRARY_PATH}"
$ ./configure --no-updates -noX11 -cuda gnu
$ make -j 40 && make install
```

## 编译 MPI+CUDA 版本

- 安装 Amber18 的 CUDA+mpi 版本
```bash
$ ./configure --no-updates -noX11 -cuda -mpi gnu
$ make -j 40 && make install
```

编译完成后退出计算节点
```bash
$ exit
```


## 参考文献

- [Amber 官网](https://ambermd.org/)
