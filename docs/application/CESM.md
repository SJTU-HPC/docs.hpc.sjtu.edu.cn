# <center>CESM</center>

------

CESM （Community Earth System Model）是一种全耦合的全球气候模型，它提供了地球过去，现在和将来的气候状态的计算机模拟。

## 获得CESM代码

CESM是一种免费的开源软件，但是在授予对Subversion存储库的访问权限之前，需要[注册](http://www.cesm.ucar.edu/models/register/register.html)。 
注册完成后，可以将CESM项目下载至<cesm-base>：

```shell
$ cd <cesm-base>
$ svn co https://svn-ccsm-models.cgd.ucar.edu/cesm1/release_tags/cesm1_2_2_1
```

## 构建CESM的依赖环境

载入Intel编译环境，进行依赖构建：

```shell
$ srun -N 1 -p cpu --exclusive --pty /bin/bash
$ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
```

定义相关的环境变量：

```shell
$ export CESM_ROOT=<cesm-base>
$ export CESM_ENV=$CESM_ROOT/env
$ export CESM_ENVSRC=$CESM_ROOT/src
$ export PATH=$CESM_ENV/bin:$PATH
$ export LD_LIBRARY_PATH=$CESM_ENV/lib:$LD_LIBRARY_PATH
```

分别进行 `curl,hdf5,netcdf,pnercdf` 的编译和安装：

```shell
$ cd $CESM_ENVSRC
$ wget https://github.com/curl/curl/releases/download/curl-7_69_1/curl-7.69.1.tar.bz2
$ tar xvf ./curl-7.69.1.tar.bz2
$ ./buildconf
$ CC=icc CXX=icpc F77=ifort F90=ifort FC=ifort ./configure --prefix=$CESM_ENV
$ make -j40
$ make -j40 install
```

```shell
$ wget http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.bz2
$ tar xvf ./hdf5-1.10.6.tar.bz2
$ CC=icc CXX=icpc F77=ifort F90=ifort FC=ifort ./configure --prefix=$CESM_ENV --enable-fortran
$ make -j40
$ make -j40 install
```

```shell
$ wget https://github.com/Unidata/netcdf-c/archive/v4.7.3.tar.gz
$ tar xvf ./v4.7.3.tar.gz
$ CC=icc CPPFLAGS=-I$CESM_ENV/include CXX=icpc F77=ifort F90=ifort FC=ifort LDFLAGS=-L$CESM_ENV/lib ./configure --prefix=$CESM_ENV --enable-netcdf-4
$ make -j40
$ make -j40
```

```shell
$ wget https://github.com/Unidata/netcdf-cxx4/archive/v4.3.1.tar.gz
$ tar xvf ./v4.3.1.tar.gz
$ CC=icc CPPFLAGS=-I$CESM_ENV/include CXX=icpc F77=ifort F90=ifort FC=ifort LDFLAGS=-L$CESM_ENV/lib ./configure --prefix=$CESM_ENV
$ make -j40
$ make -j40 install
```

```shell
$ wget https://github.com/Unidata/netcdf-fortran/archive/v4.5.2.tar.gz
$ tar xvf ./v4.5.2.tar.gz
$ CC=icc CPPFLAGS=-I$CESM_ENV/include CXX=icpc F77=ifort F90=ifort FC=ifort LDFLAGS=-L$CESM_ENV/lib ./configure --prefix=$CESM_ENV
$ make -j40
$ make -j40 install
```

```shell
$ wget https://parallel-netcdf.github.io/Release/pnetcdf-1.12.1.tar.gz
$ tar xvf ./pnetcdf-1.12.1.tar.gz
$ CC=mpiicc CXX=mpiicpc F77=mpiifort F90=mpiifort FC=mpiifort ./configure --prefix=$CESM_ENV --enable-shared
$ make -j40
$ make -j40 install
```

安装`perl`的依赖模块：

```shell
$ module load perl
$ cpan install Switch
$ cpan install XML::LibXML
```

## CESM算例构建

```shell
$ cd cesm1_2_2_1/scripts
$ cd ccsm_utils/Machines/
$ touch mkbatch.pi && chmod +x ./mkbatch.pi
$ vim config_machines.xml
```

在`config_machines.xml`中增加配置`pi`：

```xml
<machine MACH="pi">
    <DESC>config for sjtu pi</DESC>                                 <!-- can be anything -->
    <OS>LINUX</OS>                              <!-- LINUX,Darwin,CNL,AIX,BGL,BGP -->
    <COMPILERS>intel,ibm,pgi,pathscale,gnu,cray,lahey</COMPILERS>     <!-- intel,ibm,pgi,pathscale,gnu,cray,lahey -->
    <MPILIBS>openmpi,mpich,mpt,mpt,ibm,mpi-serial,impi</MPILIBS>                <!-- openmpi, mpich, ibm, mpi-serial -->
    <MPILIB>impi</MPILIB>
    <RUNDIR>$EXEROOT/../run</RUNDIR>                       <!-- complete path to the run directory -->
    <EXEROOT>USERDEFINED_required_build</EXEROOT>                     <!-- complete path to the build directory -->
    <DIN_LOC_ROOT>USERDEFINED_required_build</DIN_LOC_ROOT>           <!-- complete path to the inputdata directory -->
    <DIN_LOC_ROOT_CLMFORC>USERDEFINED_optional_build</DIN_LOC_ROOT_CLMFORC> <!-- path to the optional forcing data for CLM (for CRUNCEP forcing) -->
    <DOUT_S>TRUE</DOUT_S>                                            <!-- logical for short term archiving -->
    <DOUT_S_ROOT>$EXEROOT/../archive</DOUT_S_ROOT>               <!-- complete path to a short term archiving directory -->
    <DOUT_L_MSROOT>$EXEROOT/../l_archive</DOUT_L_MSROOT>           <!-- complete path to a long term archiving directory -->
    <CCSM_BASELINE>USERDEFINED_optional_run</CCSM_BASELINE>           <!-- where the cesm testing scripts write and read baseline results -->
    <CCSM_CPRNC>USERDEFINED_optional_test</CCSM_CPRNC>                <!-- path to the cprnc tool used to compare netcdf history files in testing -->
    <BATCHQUERY>USERDEFINED_optional_run</BATCHQUERY>
    <BATCHSUBMIT>csh</BATCHSUBMIT>
    <SUPPORTED_BY>USERDEFINED_optional</SUPPORTED_BY>
    <GMAKE_J>8</GMAKE_J>
    <MAX_TASKS_PER_NODE>40</MAX_TASKS_PER_NODE>
</machine>
```

使用`create_newcase`构建算例：

```shell
$ cd <cesm-base>/cesm1.2.2.1/scripts/
$ ./create_newcase -case ../cases/lbtest-f19_g16-B -res f19_g16 -compset B -mach pi
```

配置和编译该算例：

```shell
$ cd <cesm-base>/cesm1.2.2.1/cases/lbtest-f19_g16-B
$ ./cesm_setup
$ ./lbtest-f19_g16-B.build
```

## 参考文献

- [CESM官方网站](https://http://www.cesm.ucar.edu/)
- [CESM User Guide](http://www.cesm.ucar.edu/models/cesm1.2/cesm/doc/usersguide/book1.html)
