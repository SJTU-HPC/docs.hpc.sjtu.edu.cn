# <center>用 Conda 安装生信软件</center>
------------
本文档介绍使用 Conda 在个人目录中安装生物信息类应用软件。

- [openslide](#openslide-python)
- [pandas](#pandas)
- [cdsapi](#cdsapi)
- [STRique](#STRique)
- [r-rgl](#r-rgl)
- [sra-tools](#sra-tools)
- [DESeq2](#deseq2)
- [WGCNA](#wgcna)
- [MAKER](#maker)
- [AUGUSTUS](#augustus)

## 用 Conda 安装软件的流程 
加载 Miniconda3
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
```

创建 conda 环境来安装所需 Python 包（可指定 Python 版本，也可以不指定）
```bash
$ conda create --name mypy python=3.6
```

激活 python 环境
```bash
$ source activate mypy
```

安装之前，先申请计算节点资源（登陆节点禁止大规模编译安装）
```bash
$ srun -p small -n 4 --pty /bin/bash
```

通过 conda 安装软件包（有些软件也可以用 pip 安装。软件官网一般给出推荐，用 conda 还是 pip）
```bash
$ conda install -c bioconda openslide-python （以 openslide-python 为例）
```

## 生信软件在 Pi 上的使用：用 slurm 提交作业

Pi 上的计算，需用 slurm 脚本提交作业，或在计算节点提交交互式任务

slurm 脚本示例：申请 small 队列的 2 个核，通过 python 打印 `hello world`

```bash
#!/bin/bash
#SBATCH -J py_test
#SBATCH -p small
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH -o %j.out
#SBATCH -e %j.err

module purge
module load miniconda3/4.7.12.1-gcc-4.8.5

source activate mypy

python -c "print('hello world')"
```

我们假定以上脚本内容被写到了 `hello_python.slurm` 中，使用 `sbatch` 指令提交作业
```bash
$ sbatch hello_python.slurm
```
## 软件安装示例

许多生信软件可以在 anaconda 的 bioconda package 里找到：

[https://anaconda.org/bioconda](https://anaconda.org/bioconda)

以下为一些软件的具体安装步骤：

## openslide-python 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c bioconda openslide-python
$ conda install libiconv
```

## pandas 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c anaconda pandas
```

## cdsapi 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c conda-forge cdsapi
```

## STRique 安装

完整步骤
```bash
$ srun -p small -n 4 --pty /bin/bash
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ git clone --recursive https://github.com/giesselmann/STRique
$ cd STRique
$ pip install -r requirements.txt
$ python setup.py install 
```

## r-rgl 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c r r-rgl
```

## sra-tools 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c bioconda sra-tools
```

## DESeq2 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c bioconda bioconductor-deseq2
```
安装完成后可以在 R 中输入  `library("DESeq2")` 检测是否安装成功

## WGCNA 安装

完整步骤
```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
$ conda create -n mypy python=3.6
$ source activate mypy
$ conda install -c bioconda r-wgcna
```

## MAKER 安装

完整步骤
```bash
$ srun -p small -n 4 --pty /bin/bash
$ module purge
$ module load miniconda3
$ conda create -n mypy
$ source activate mypy
$ conda install -c bioconda maker
```

## AUGUSTUS 安装

完整步骤
```bash
$ srun -p small -n 4 --pty /bin/bash
$ module purge
$ module load miniconda3
$ conda create -n mypy
$ source activate mypy
$ conda install -c anaconda boost
$ conda install -c bioconda augustus
```

## 参考文献

- [miniconda](https://docs.conda.io/en/latest/miniconda.html)
