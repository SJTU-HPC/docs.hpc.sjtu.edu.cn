# <center>Python</center>
------------
本文档向您展示如何使用Miniconda在家目录中建立自定义的Python环境。不同的Python版本2或3，对应不同的Miniconda。

## Miniconda2
加载Miniconda2

```bash
$ module purge
$ module load miniconda2/4.6.14-gcc-4.8.5
```
创建conda环境来安装所需Python包。
```bash
$ conda create --name mypython2 numpy scipy matplotlib ipython jupyter
```
指定python版本（不指定将默认安装最新版）
```bash
$ conda create --name mypython2 python==2.7
```
激活 python 环境
```bash
$ source activate mypython2
```
通过conda或pip添加更多软件包

```bash
$ conda install YOUR_PACKAGE
$ pip install YOUR_PACKAGE
```

## Miniconda 3
加载Miniconda3
```bash
$ module purge
$ module load miniconda3/4.6.14-gcc-4.8.5
```
创建conda环境来安装所需Python包。
```bash
$ conda create --name mypython3 numpy scipy matplotlib ipython jupyter
```
激活 python 环境
```bash
$ source activate mypython3
```
通过conda或pip添加更多软件包
```bash
$ conda install YOUR_PACKAGE
$ pip install YOUR_PACKAGE
```

## 使用全局预创建的conda环境

集群已创建全局的conda环境，该环境主要面向生物医学用户主要包含tensorflow-gpu@2.0.0，R@3.6.1，python@3.7.4 。使用以下指令激活环境：

```bash
$ module load miniconda3/4.6.14-gcc-4.8.5 
$ source activate /lustre/opt/condaenv/life_sci
```

conda拓展模块查询方法
```bash
$ conda list
```

R拓展模块查询方法
```bash
$ R
> installed.packages()
```

## 使用Miniconda向slurm提交作业

以下为python示例作业脚本，我们将向slurm申请两cpu核心，并在上面通过python打印`hello world`。

```bash
#!/bin/bash
#SBATCH -J hello-python
#SBATCH -p small
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 2

module purge
module load miniconda3/4.6.14-gcc-4.8.5

source activate mypython3

python -c "print('hello world')"
```

我们假定以上脚本内容被写到了`hello_python.slurm`中，使用`sbatch`指令提交作业。

```bash
$ sbatch hello_python.slurm
```



## 参考文献

- [miniconda](https://docs.conda.io/en/latest/miniconda.html)
