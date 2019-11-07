# <center>VASP<center/>

-------
## 编译VASP

1.解压缩VASP
```
$ tar xvf vasp.5.4.4.tar.bz2
$ cd vasp.5.4.4
```
2.设置VTST
```
$ wget http://theory.cm.utexas.edu/code/vtstcode.tgz
$ tar xvf vtstcode.tgz
```
3.在VASP中备份文件（可选）
```
$ cp src/chain.F src/chain.F-org
```
4.在VASP中替换文件
```
$ cp vtstcode-171/* src/
```
5.修改源代码
在src/main.F中将第3146行如下内容：
```
CALL CHAIN_FORCE(T_INFO%NIONS,DYN%POSION,TOTEN,TIFOR, &
     LATT_CUR%A,LATT_CUR%B,IO%IU6)
```
修改为：
```
CALL CHAIN_FORCE(T_INFO%NIONS,DYN%POSION,TOTEN,TIFOR, &
     TSIF,LATT_CUR%A,LATT_CUR%B,IO%IU6)
!     LATT_CUR%A,LATT_CUR%B,IO%IU6)
```
在src/.objects中chain.o（第72行）之前添加如下内容：
```
bfgs.o dynmat.o instanton.o lbfgs.o sd.o cg.o dimer.o bbm.o \
fire.o lanczos.o neb.o qm.o opt.o \
```

!!! tip
    注意后面没有空格

6.加载 icc 和 impi
```
$ module load icc/16.0
$ module load impi/2016
```
7.设置MKLROOT变量
检查ifort路径，设置MKLROOT
```
$ which ifort
/lustre/spack/tools/linux-centos7-x86_64/intel-16.0.4/intel-parallel-studio-cluster.2016.4-ybjjq75tqpzgzjc4drolyijzm45g5qul/compilers_and_libraries_2016.4.258/linux/bin/intel64/ifort
$ export MKLROOT=/lustre/spack/tools/linux-centos7-x86_64/intel-16.0.4/intel-parallel-studio-cluster.2016.4-ybjjq75tqpzgzjc4drolyijzm45g5qul/compilers_and_libraries_2016.4.258/linux/mkl
```
8.使用arch / makefile.include.linux_intel作为模板
```
$ cp arch/makefile.include.linux_intel makefile.include
```
9.清理之前编译的文件
```
$ make veryclean
```
10.编译
```
$ make
```
现在bin中的二进制文件包含vasp_std vasp_gam vasp_ncl

## 参考文献
* [VASP 5.4.1+VTST编译安装](http://hmli.ustc.edu.cn/doc/app/vasp.5.4.1-vtst.htm)
* [VTST installation](http://theory.cm.utexas.edu/vtsttools/installation.html)


