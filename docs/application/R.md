# <center>R</center>

-----------

本文档向您展示如何使用Miniconda在家目录中建立自定义的R环境。

## 使用Miniconda 3环境安装R

加载Miniconda 3

```bash
$ module purge
$ module load miniconda3/4.7.12.1-gcc-4.8.5
```

创建conda环境

```bash
$ conda create --name R
```

激活R环境

```bash
$ source activate R
```

安装R

```bash
$ conda install -c r r==3.6.0
```

通过conda（例如RMySQL）添加更多软件包。

```bash
$ conda install -c conda-forge r-rmysql
```

您可以在[https://anaconda.org/](https://anaconda.org/)中搜索安装命令。

## 使用conda环境中R提交slurm作业

使用conda环境中R运行单节点作业脚本示例r_conda.slurm如下：

```bash
#!/bin/bash

#SBATCH -J R
#SBATCH -p small
#SBATCH -n 1
#SBATCH -o %j.out
#SBATCH -e %j.err

source /usr/share/Modules/init/bash

module purge
module load miniconda3/4.7.12.1-gcc-4.8.5
source activate R

R hello.r
```

并使用如下指令提交：

```bash
$ sbatch r_conda.slurm
```

## 在R交互终端中安装R模块
```bash
$ R --version
R version 3.6.1 (2019-07-05) -- "Action of the Toes"
Copyright (C) 2019 The R Foundation for Statistical Computing
Platform: x86_64-conda_cos6-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under the terms of the
GNU General Public License versions 2 or 3.
For more information about these matters see
https://www.gnu.org/licenses/.

$ R
R version 3.6.1 (2019-07-05) -- "Action of the Toes"
Copyright (C) 2019 The R Foundation for Statistical Computing
Platform: x86_64-conda_cos6-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> if (!requireNamespace("BiocManager", quietly = TRUE))
+     install.packages("BiocManager")
> BiocManager::install()
Bioconductor version 3.10 (BiocManager 1.30.10), R 3.6.1 (2019-07-05)
Old packages: 'boot', 'cluster', 'foreign', 'KernSmooth', 'MASS', 'mgcv',
  'nlme', 'survival'               
...
```
