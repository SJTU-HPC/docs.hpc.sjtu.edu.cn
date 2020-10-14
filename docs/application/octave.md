# <center>Octave</center>

---------

## 简介

GNU Octave 是一种采用高级编程语言的主要用于数值分析的软件。Octave 有助于以数值方式解决线性和非线性问题，并使用与 MATLAB 兼容的语言进行其他数值实验。它也可以作为面向批处理的语言使用。因为它是 GNU 计划的一部分，所以它是 GNU 通用公共许可证条款下的自由软件。


## Pi 上的 Octave

查看 Pi 上已编译的软件模块:
```bash
module av octave
```

调用该模块:
```bash
module load octave/5.2.0
```

示例 slurm 脚本：在 small 队列上，总共使用 4 核 (n = 4)，脚本名称设为 slurm.test

!!! example "small 队列 slurm 脚本示例 Octave"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=test          # 作业名
    #SBATCH --partition=small        # small 队列
    #SBATCH -n 4                     # 总核数 4
    #SBATCH --ntasks-per-node=4      # 每节点核数
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load octave/5.2.0

    ulimit -s unlimited
    ulimit -l unlimited

    octave [FILE_NAME]
    ```

用下方语句提交作业
```bash
sbatch slurm.test
```


## 使用 HPC Studio 启动 Octave 可视化界面

首先参照[可视化平台](../../login/HpcStudio/)开启远程桌面，并在远程桌面中启动终端，并输入以下指令：

```bash
module load octave/5.2.0
octave [FILE_NAME]
```

## 参考文献

- [Octave 官方网站](https://www.gnu.org/software/octave/)

