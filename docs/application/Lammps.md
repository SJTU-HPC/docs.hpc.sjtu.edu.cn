# <center>Lammps</center> 

-----

## 加载预安装的 Lammps

Pi 2.0 系统中已经预装 lammps/20190807 (Intel+cpu 版本)，可以用以下命令加载: 

```
$ module load lammps/20190807-intel-19.0.5-impi
```

## 提交 Intel+CPU 版本 Lammps 任务

使用 Intel 2019 编译的 CPU 版本 Lammps 运行单节点作业脚本示例 lammps_cpu_intel.slurm：


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
module load lammps/20190807-intel-19.0.5-impi

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun --mpi=pmi2 lmp -i YOUR_INPUT_FILE
```

并使用如下指令提交：

```bash
$ sbatch lammps_cpu_intel.slurm
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

srun --mpi=pmi2 ~/lammps-3Mar20/src/lmp_intel_cpu_intelmpi -i YOUR_INPUT_FILE
```





