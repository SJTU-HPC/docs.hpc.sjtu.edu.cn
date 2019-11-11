# <center>Python</center>
------------
本文档向您展示如何使用Miniconda在家目录中建立自定义的Python环境。不同的Python版本2或3，对应不同的Miniconda。

## MINICONDA2
加载MINICONDA2

```
$ module purge
$ module load miniconda2/4.6.14-gcc-4.8.5
```
创建conda环境来安装所需Python包。
```
$ conda create --name mypython2 numpy scipy matplotlib ipython jupyter
```
激活 python 环境
```
$ source activate mypython2
```
通过conda或pip添加更多软件包

```
$ conda install YOUR_PACKAGE
$ pip install YOUR_PACKAGE
```

## Miniconda 3
加载Miniconda3
```
$ module purge
$ module load miniconda3/4.6.14-gcc-4.8.5
```
创建conda环境来安装所需Python包。
```
$ conda create --name mypython3 numpy scipy matplotlib ipython jupyter
```
激活 python 环境
```
$ source activate mypython3
```
通过conda或pip添加更多软件包
```
$ conda install YOUR_PACKAGE
$ pip install YOUR_PACKAGE
```

## 参考文献

- [miniconda](https://docs.conda.io/en/latest/miniconda.html)