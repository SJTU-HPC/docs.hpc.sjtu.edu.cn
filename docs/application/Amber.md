# <center>AMBER<center/>

-------
## 安装Amber14 MPI版本
Amber14[下载](http://ambermd.org/Amber14-get.htmlTools)，将下载的文件上传到Pi集群。
加载基本模块解压缩文件
```
$ module purge
$ module load gcc/5.4 openmpi/3.1
$ tar -xvjf AmberTools14.tar.bz2                 
$ tar -xvjf Amber14.tar.bz2
```
配置Amber14.
```
$ cd amber14
$ export AMBERHOME=$HOME/amber14  
$ ./configure -mpi -noX11 gnu
```
安装Amber14.
```
$ make install
```
使用Amber 14, 输入amber.sh.
```
$ source amber.sh
```
## 安装Amber14 GPU 版本 
Amber14下载[http://ambermd.org/Amber14-get.htmlTools]，AmberTools14下载[http://ambermd.org/AmberTools14-get.html]，将下载的文件上传到Pi集群。
加载基本模块解压缩文件
```
$ module purge
$ module load gcc/4.8 openmpi/3.1 cuda/6.5
$ tar -xvjf AmberTools14.tar.bz2                 
$ tar -xvjf Amber14.tar.bz2
```
配置Amber14.
```
$ cd amber14
$ export AMBERHOME=$HOME/amber14  
$ ./configure -cuda -mpi -noX11 gnu
```
安装Amber14.
```
$ make install
```
使用Amber 14, 输入amber.sh.
```
$ source amber.sh
```
## 参考文献
* [Amber 14 Reference Manual Chapter 2](http://ambermd.org/doc12/Amber14.pdf)
