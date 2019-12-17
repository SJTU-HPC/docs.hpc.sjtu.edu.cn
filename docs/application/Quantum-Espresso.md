# Quantum-Espresso

--------

## 加载预安装的Quantum-Espresso

Pi2.0 系统中已经预装 Quantum-Espresso 6.4.1 (Intel+CPU 版本)，可以用以下命令加载: 

```bash
module load quantum-espresso/6.4.1-intel-19.0.4-impi
```

## 提交Intel+CPU版本Quantum-Espresso任务

加载后即可使用 `pw.x` 等命令。

使用Intel编译的CPU版本Quantum-Espresso运行单节点作业脚本示例qe_cpu_intel.slurm如下：

```bash
#!/bin/bash

#SBATCH -J qe_test
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH -o %j.out
#SBATCH -e %j.err

ulimit -s unlimited
ulimit -l unlimited

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi

module purge
module load quantum-espresso/6.4.1-intel-19.0.4-impi

srun pw.x < rlx.in
```

其中 rlx.in 是您提供的参数文件.


并使用如下指令提交：

```bash
$ sbatch qe_cpu_intel.slurm
```
