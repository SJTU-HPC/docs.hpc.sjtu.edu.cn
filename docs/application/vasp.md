# <center>VASP<center/>

-------
## 编译 Intel+CPU 版本 VASP

- 解压缩 VASP
```bash
$ tar xvf vasp.5.4.4.tar.bz2
$ cd vasp.5.4.4
```
- 如果需要 VTST 拓展，使用以下方式安装 (不同的 VTST 和 VASP 版本可能有不同的安装方式)
  - 从官网下载
```bash
$ wget http://theory.cm.utexas.edu/code/vtstcode-179.tgz
$ tar xvf vtstcode.tgz
$ cd vasp.5.4.4   (再回到 vasp 文件夹进行安装)
```
  - 在 vasp 文件夹里开始安装。首先替换部分 VASP 文件：
```bash
$ cp ../vtstcode-179/* src/
```
  - 修改源文件, 在 `src/main.F` 中将第3146行如下内容：
```fortran
CALL CHAIN_FORCE(T_INFO%NIONS,DYN%POSION,TOTEN,TIFOR, &
     LATT_CUR%A,LATT_CUR%B,IO%IU6)
```
修改为：
```fortran
CALL CHAIN_FORCE(T_INFO%NIONS,DYN%POSION,TOTEN,TIFOR, &
     TSIF,LATT_CUR%A,LATT_CUR%B,IO%IU6)
!     LATT_CUR%A,LATT_CUR%B,IO%IU6)
```
在 src/.objects中chain.o（第72行）之前添加如下内容：
```bash
bfgs.o dynmat.o instanton.o lbfgs.o sd.o cg.o dimer.o bbm.o \
fire.o lanczos.o neb.o qm.o opt.o \
```

!!! tip
    注意后面没有空格

- 加载 intel 编译器，对于 VASP 5.4.4，我们推荐 intel-parallel-studio-2018
```bash
$ module load intel-parallel-studio/cluster.2018.4-intel-18.0.4
```
上述操作后会 load 包括 intel compilers, intel-mpi, intel-mkl 等所需的编译器组件，您可以使用 ``echo $MKLROOT`` 等方式检查是否成功导入.

- 使用 `arch/makefile.include.linux_intel` 作为模板
```bash
$ cp arch/makefile.include.linux_intel makefile.include
```

- 清理之前编译的文件（某些情况需要）并编译
```bash
$ make veryclean
$ make     （不建议用 -j）
```
现在 `./bin` 目录中的二进制文件包含 vasp_std vasp_gam vasp_ncl. 您也可以单独编译每一个，用指令例如：`make std` 即可编译 vasp_std

## 提交 Intel+CPU 版本 VASP 任务

使用intel编译的CPU版本VASP运行单节点作业脚本示例vasp_cpu_intel.slurm如下：

```bash
#!/bin/bash

#SBATCH -J vasp_test
#SBATCH -p cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load intel-parallel-studio/cluster.2018.4-intel-18.0.4

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:tmi

ulimit -s unlimited
ulimit -l unlimited

srun /path/to/your_vasp_dir/bin/vasp_std
```

并使用如下指令提交：

```bash
$ sbatch vasp_cpu_intel.slurm
```

## 编译 Intel+GPU 版本 VASP

GPU 需要在 dgx2 队列上编译，以如下命令申请计算节点资源用于编译：
```bash
$ srun -p dgx2 --gres=gpu:1 --ntasks-per-node 6 --pty /bin/bash
```

- 编译 GPU 版本需要首先编译CPU版本，在其基础上使用下述命令

```bash
$ # 修改 makefile.include 中的 CUDA_ROOT 路径为 CUDA_ROOT  := $(CUDA_HOME)
$ # 修改 makefile.include 中的 -openmp 参数为 -qopenmp
$ module load cuda/10.2.89-intel-19.0.4
$ module load intel-parallel-studio/cluster.2018.4-intel-18.0.4
$ make gpu
```

## 提交 Intel+GPU 版本 VASP 任务

使用 intel 编译的 GPU 版本 VASP 运行单卡作业脚本示例 vasp_gpu_intel.slurm 如下：

注意：INCAR 里倒数第三行需要改为 LREAL = .TRUE.

```bash
#!/bin/bash
#SBATCH -J vasp_gpu
#SBATCH -p dgx2
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

module purge
module load intel-parallel-studio/cluster.2018.4-intel-18.0.4
module load cuda/10.2.89-intel-19.0.4

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

ulimit -s unlimited
ulimit -l unlimited

srun /path/to/your_vasp_dir/bin/vasp_gpu
```

并使用如下指令提交：

```bash
$ sbatch vasp_gpu_intel.slurm
```

## VASP 算例
- [Pt](https://hpc.sjtu.edu.cn/Item/docs/mp-126_Pt.tar.gz)


## 参考文献

- [VASP 5.4.1+VTST编译安装](http://hmli.ustc.edu.cn/doc/app/vasp.5.4.1-vtst.htm)
- [VTST installation](http://theory.cm.utexas.edu/vtsttools/installation.html)
