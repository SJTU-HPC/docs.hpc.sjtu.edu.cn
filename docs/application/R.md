# <center>R</center>

-----------

本文档向您展示如何使用Miniconda在家目录中建立自定义的R环境。

## 使用Miniconda 3环境安装R

加载Miniconda 3

```
$ module purge
$ module load miniconda3/4.6.14-gcc-4.8.5
```

创建conda环境

```
$ conda create --name R
```

激活R环境

```
$ source activate R
```

安装R

```
conda install -c r r==3.6.0
```

通过conda（例如RMySQL）添加更多软件包。

```
$ conda install -c conda-forge r-rmysql
```

您可以在[https://anaconda.org/](https://anaconda.org/)中搜索安装命令。

## 使用conda环境中R提交slurm作业

使用conda环境中R运行单节点作业脚本示例r_conda.slurm如下：

```
#!/bin/bash

#SBATCH -J R
#SBATCH -p small
#SBATCH -n 1
#SBATCH -o %j.out
#SBATCH -e %j.err

source /usr/share/Modules/init/bash

module purge
module load miniconda3/4.6.14-gcc-4.8.5
source activate R

R hello.r
```

并使用如下指令提交：

```
$ sbatch r_conda.slurm
```

## 在R交互终端中安装R模块
```
$ R --version
R version 3.5.0 (2018-04-23) -- "Joy in Playing"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)
> R
> source("https://bioconductor.org/biocLite.R")                    
...  
Bioconductor version 3.7 (BiocInstaller 1.30.0), ?biocLite for help
> biocLite()                  
...
```
