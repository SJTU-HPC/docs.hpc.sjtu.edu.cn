# <center>VASP<center/>

-------
## 编译VASP

1. 解压缩 VASP
```
$ tar xvf vasp.5.4.4.tar.bz2
$ cd vasp.5.4.4
```
2. 如果需要 VTST 拓展，使用以下方式安装 (不同的 VTST 和 VASP 版本可能有不同的安装方式)
  * 从官网下载
```
  $ wget http://theory.cm.utexas.edu/code/vtstcode.tgz
  $ tar xvf vtstcode.tgz
```
  * 备份 VASP 文件（可选）
```
  $ cp src/chain.F src/chain.F-org
```
  * 替换部分 VASP 文件
```
  $ cp vtstcode-171/* src/
```
  * 修改源文件, 在 `src/main.F` 中将第3146行如下内容：
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

3. 加载 intel 编译器，对于 VASP 5.4.4，我们推荐 intel-parallel-studio-2018
```
$ module load intel-parallel-studio/cluster.2018.3-gcc-4.8.5
```
上述操作后会 load 包括 intel compilers, intel-mpi, intel-mkl 等所需的编译器组件，您可以使用 ``echo $MKLROOT`` 等方式检查是否成功导入.

4. 使用 `arch/ makefile.include.linux_intel` 作为模板
```
$ cp arch/makefile.include.linux_intel makefile.include
```

5. 清理之前编译的文件（某些情况需要）并编译
```
$ make veryclean ## WARNING: all previously generated files will be removed
$ make
```
现在 `./bin` 目录中的二进制文件包含 vasp_std vasp_gam vasp_ncl. 您也可以单独编译每一个，用指令例如：`make std` 即可编译 vasp_std

## 参考文献
* [VASP 5.4.1+VTST编译安装](http://hmli.ustc.edu.cn/doc/app/vasp.5.4.1-vtst.htm)
* [VTST installation](http://theory.cm.utexas.edu/vtsttools/installation.html)


