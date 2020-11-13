# Spack 环境

Spack是一个包管理工具，为大型超级计算中心设计，支持各种平台和环境上的软件的多个版本和配置。

用户可在集群中安装多种版本的软件，且可以相互隔离、无特权需求，用户可在集群中安装一个软件的不同版本来使用。



用户可以通过`spec`句法来对自己需要安装的软件来指定不同的参数。

## 安装Spack

Spack安装十分简单，您可以在github下载spack文件到本地，然后将spack环境变量导入现有窗口中。

### 下载Spack包管理器文件

```bash
cd ~
git clone https://github.com/spack/spack.git
```

### 导入Spack环境

```bash
$ . spack/share/spack/setup-env.sh
```

您也可以将Spack环境放入`.bashrc`中，这样在每次加载shell时可默认加载spack环境。

```bash
echo "source /opt/spack/share/spack/setup-env.sh" >> ~/.bashrc
source ~/.bashrc
```

### 检查安装情况

当完成安装和导入spack步骤时，您可以使用`spack`命令来测试功能。

本次测试使用`spec`模块来做测试，`spec`可查询安装此软件包需要的依赖项。

下列命令为使用现有`gcc`版本为`7.5.0`的编译器来编译`gcc-10.2.0`版本，如可以出现下面类似输出，则说明spack环境搭建完成。

```bash
$ spack spec gcc@10.2.0%gcc@7.5.0
Input spec
--------------------------------
gcc@10.2.0%gcc@7.5.0

Concretized
--------------------------------
gcc@10.2.0%gcc@7.5.0~binutils~bootstrap~nvptx~piclibs~strip languages=c,c++,fortran patches=2c18531a23623596e1daf6f0dd963cf72c208945ecad90515640c3ab23991159 arch=linux-ubuntu18.04-skylake
    ^gmp@6.1.2%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
        ^autoconf@2.69%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
            ^m4@1.4.18%gcc@7.5.0+sigsegv patches=3877ab548f88597ab2327a2230ee048d2d07ace1062efe81fc92e91b7f39cd00,fc9b61654a3ba1a8d6cd78ce087e7c96366c290bc8d2c299f09828d793b853c8 arch=linux-ubuntu18.04-skylake
                ^libsigsegv@2.12%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
            ^perl@5.30.3%gcc@7.5.0+cpanm+shared+threads arch=linux-ubuntu18.04-skylake
                ^berkeley-db@18.1.40%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
                ^gdbm@1.18.1%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
                    ^readline@8.0%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
                        ^ncurses@6.2%gcc@7.5.0~symlinks+termlib arch=linux-ubuntu18.04-skylake
                            ^pkgconf@1.7.3%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
        ^automake@1.16.2%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
        ^libtool@2.4.6%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
    ^isl@0.21%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
    ^mpc@1.1.0%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
        ^mpfr@4.0.2%gcc@7.5.0 patches=3f80b836948aa96f8d1cb9cc7f3f55973f19285482a96f9a4e1623d460bcccf0 arch=linux-ubuntu18.04-skylake
            ^autoconf-archive@2019.01.06%gcc@7.5.0 arch=linux-ubuntu18.04-skylake
    ^zlib@1.2.11%gcc@7.5.0+optimize+pic+shared arch=linux-ubuntu18.04-skylake
    ^zstd@1.4.5%gcc@7.5.0+pic arch=linux-ubuntu18.04-skylake

```

## 编译器配置

!!! info "default"
	默认情况下，spack在第一次加载环境时会自动发现当前可用的编译器



### `spack compilers`

您可以通过运行`spack compilers` 或者 `spack compiler list`来查看Spack发现的编译器

```bash
$ spack compilers                                               
==> Available compilers
-- gcc ubuntu18.04-x86_64 ---------------------------------------
gcc@10.2.0  gcc@7.5.0  gcc@5.4.0

```



### `spack compiler find `

您可以通过运行`spack compiler find`或者`spack compiler add`来自动发现或手动添加编译器路径。

```bash
$ ml load hpc_sdk
$ spack compiler find
==> Added 1 new compiler to /home/nfs/admin0/user/.spack/linux/compilers.yaml
    nvhpc@20.9
==> Compilers are defined in the following files:
    /home/nfs/admin0/user/.spack/linux/compilers.yaml
```



### `spack compiler info`

如果你想查看详细的编译器信息，您可以使用如下命令。

```bash
$ spack compiler info nvhpc                                                                                                                                                                                     
nvhpc@20.9:
        paths:
                cc = /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvc
                cxx = /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvc++
                f77 = /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvfortran
                fc = /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvfortran
        modules  = []
        operating system  = ubuntu18.04

```

如果有多个版本的软件包，您可以在软件包名称后添加`@`,例如`spack compiler info gcc@10.2.0`



### 手动配置可用编译器（可选）

如果您有兴趣手动指定`cc|cxx|f77|fc`，您可以以如下两种方式来配置`compilers.yaml`

- EDIT `~/.spack/linux/compilers.yaml`
- `spack config edit compilers`

编译器配置文件如下所示:

```bash
- compiler:
    spec: nvhpc@20.9
    paths:
      cc: /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvc
      cxx: /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvc++
      f77: /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvfortran
      fc: /home/nfs/admin0/apps/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvfortran
    flags: {}
    operating_system: ubuntu18.04
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
```



### 指定默认编译器版本（可选）

在您安装新软件包时，spack需要使用一个编译器来编译环境，如果您对编译器版本有要求，您可以通过以下两种方式来指定编译器：

- EDIT `~/.spack/package.yaml`
- `spack config edit package`

包配置文件如下所示：

```bash
packages:
  all:
    compiler: [gcc@10.2.0]                       
```



## 基础使用教程

`spack`有许多子命令可以使用，您需要了解一些基础 命令以便更好的使用spack来安装和使用软件

例如，我们要查找当前spack环境中安装的软件包，可以使用`spack find`

```bash
$ spack find                                                                                                                                                                                                    
==> 93 installed packages

-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
autoconf@2.69                bzip2@1.0.8    gcc@7.5.0     gmp@6.1.2      libedit@3.1-20191231  libtool@2.4.6        lua@5.3.5                  mpc@1.1.0   ncurses@6.2     perl@5.30.3    tcl@8.6.10   zstd@1.4.5
autoconf-archive@2019.01.06  curl@7.72.0    gdbm@1.18.1   isl@0.18       libiconv@1.16         libunistring@0.9.10  lua-luafilesystem@1_7_0_2  mpc@1.1.0   openssh@8.4p1   pkgconf@1.7.3  unzip@6.0
automake@1.16.2              diffutils@3.7  gettext@0.21  isl@0.21       libidn2@2.3.0         libxml2@2.9.10       lua-luaposix@33.4.0        mpfr@3.1.6  openssl@1.1.1h  readline@8.0   xz@5.2.5
berkeley-db@18.1.40          expat@2.2.9    git@2.29.0    libbsd@0.10.0  libsigsegv@2.12       lmod@8.3             m4@1.4.18                  mpfr@4.0.2  pcre2@10.35     tar@1.32       zlib@1.2.11

```



### 列出spack可安装的软件包

当您准备安装某一个软件时，需要先确认软件包是否存在于spack软件包列表中。您可以使用`spack list`命令来查找可用软件，或者可以打开[Package List](https://spack.readthedocs.io/en/latest/package_list.html#package-list)网页来查找软件包信息。





#### `spack list`

此命令可以查询spack可安装的软件包,截至2020年11月11日，目前可用软件包为5021个。



```bash
 
$ spack list                                                                                                                  
==> 5021 packages.                                                                                                                                                                                                 
3proxy                     icu4c                            perl-set-intervaltree                     py-pyfftw                                       r-popgenome                                                  
abduco                     id3lib                           perl-set-intspan                          py-pyfits                                       r-popvar                                                     
abi-compliance-checker     idba                             perl-set-scalar                           py-pyflakes                                     r-powerlaw                                                   
abi-dumper                 idl                              perl-soap-lite                            py-pygdal                                       r-prabclus                                                   
abinit                     iegenlib                         perl-star-fusion                          py-pygdbmi                                      r-praise                                                     
abseil-cpp                 ignite                           perl-statistics-basic                     py-pygelf                                       r-preprocesscore                                             
abyss                      igraph                           perl-statistics-descriptive               py-pygit2                                       r-prettyunits                                                
accfft                     igv                              perl-statistics-pca                       py-pyglet                                       r-processx    
......
```

在执行`spack list`命令时，我们隐藏了一些包文件。您可以使用`spack list`命令来查找可用软件，或者可以打开[Package List](https://spack.readthedocs.io/en/latest/package_list.html#package-list)网页来查找软件包信息。

如果想寻找可安装包中的名称来匹配您想要安装的软件，可以使用`*`或`？`进行匹配。

`*`代表开始和结束的匹配，`mpi`意思和`*mpi*`相同;

下列示例为查找mpi相关的软件包

```bash
$ spack list mpi
==> 23 packages.
compiz       intel-mpi             mpi-bash  mpifileutils  mpileaks  mpir               openmpi  phylobayesmpi  py-mpi4py  rempi         sst-dumpi  vampirtrace
fujitsu-mpi  intel-mpi-benchmarks  mpich     mpilander     mpip      mpix-launch-swift  pbmpi    pnmpi          r-rmpi     spectrum-mpi  umpire

```





#### `spack info`

当您已经明确需要安装的软件包时，此时您可以使用`spack info <package>`来查看此软件包的更多信息。

??? "info"
    ```bash
    [~] spack info openmpi                                                                                                                                                                                             
    AutotoolsPackage:   openmpi                                                                                                                                                                                        

    Description:                                                                                                                                                                                                       
        An open source Message Passing Interface implementation. The Open MPI                                                                                                                                          
        Project is an open source Message Passing Interface implementation that                                                                                                                                        
        is developed and maintained by a consortium of academic, research, and                                                                                                                                         
        industry partners. Open MPI is therefore able to combine the expertise,                                                                                                                                        
        technologies, and resources from all across the High Performance                                                                                                                                               
        Computing community in order to build the best MPI library available.                                                                                                                                          
        Open MPI offers advantages for system and software vendors, application                                                                                                                                        
        developers and computer science researchers.                                                                                                                                                                   
    
    Homepage: http://www.open-mpi.org                                                                                                                                                                                  
    
    Maintainers: @hppritcha                                                                                                                                                                                            
    
    Tags:                                                                                                                                                                                                              
        None                                                                                                                                                                                                           
    
    Preferred version:                                                                                                                                                                                                 
        3.1.6     http://www.open-mpi.org/software/ompi/v3.1/downloads/openmpi-3.1.6.tar.bz2                                                                                                                           
    
    Safe versions:                                                                                                                                                                                                     
        master    [git] https://github.com/open-mpi/ompi.git on branch master                                                                                                                                          
        4.0.5     http://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.5.tar.bz2                                                                                                                           
        4.0.4     http://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.4.tar.bz2                                                                                                                           
        4.0.3     http://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.3.tar.bz2                                                                                                                           
        4.0.2     http://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.2.tar.bz2                                                                                                                           
        4.0.1     http://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.1.tar.bz2
    
        ......
    
    Variants:
        Name [Default]           Allowed values          Description
        =====================    ====================    ==============================================================================
    
        atomics [off]            on, off                 Enable built-in atomics
        cuda [off]               on, off                 Enable CUDA support
        cxx [off]                on, off                 Enable C++ MPI bindings
        cxx_exceptions [off]     on, off                 Enable C++ Exception support
        fabrics [none]           none, auto, fca,        List of fabrics that are enabled; 'auto' lets openmpi determine
                                 knem, hcoll, verbs,
                                 psm2, psm, xpmem,
                                 ucx, mxm, cma, ofi
        gpfs [on]                on, off                 Enable GPFS support (if present)
        java [off]               on, off                 Build Java support
    
        ......
    
    Installation Phases:
        autoreconf    configure    build    install
    
    Build Dependencies:
        autoconf  binutils  hcoll  java  libfabric  lsf     m4   numactl   openpbs  pkgconfig  singularity  sqlite  valgrind  zlib
        automake  fca       hwloc  knem  libtool    lustre  mxm  opa-psm2  perl     rdma-core  slurm        ucx     xpmem
    
    Link Dependencies:
        binutils  fca  hcoll  hwloc  java  knem  libfabric  lsf  lustre  mxm  numactl  opa-psm2  openpbs  rdma-core  singularity  slurm  sqlite  ucx  valgrind  xpmem  zlib
    
    Run Dependencies:
        None
    
    Virtual Packages: 
        openmpi@2.0.0: provides mpi@:3.1
        openmpi@1.7.5: provides mpi@:3.0
        openmpi@1.6.5 provides mpi@:2.2
        openmpi provides mpi
    
    ```

通过查询软件包信息，您可以了解可添加的编译项以便更好的编译需要的软件。





#### `spack versions`

通过使用`spack versions`，您可以找到某一个软件包的可用版本。

这里以安装`openmpi`为例，我们可以看到openmpi的版本号，我们可以找到需要的版本号来执行`install`的操作。

输出中`Safe version`代表spack已经校验过sum值的，不会在安装时遭受黑客攻击和篡改；

`Remote version`为未校验过sum值的软件包。

```bash
$ spack versions openmpi                                                                                                                                                                                        
==> Safe versions (already checksummed):
  master  4.0.2  3.1.5  3.1.1  3.0.3  2.1.6  2.1.2  2.0.3  1.10.7  1.10.3  1.8.8  1.8.4  1.8    1.7.2  1.6.4  1.6    1.5.2  1.4.4  1.4    1.3.1  1.2.7  1.2.3  1.1.5  1.1.1  1.0
  4.0.5   4.0.1  3.1.4  3.1.0  3.0.2  2.1.5  2.1.1  2.0.2  1.10.6  1.10.2  1.8.7  1.8.3  1.7.5  1.7.1  1.6.3  1.5.5  1.5.1  1.4.3  1.3.4  1.3    1.2.6  1.2.2  1.1.4  1.1
  4.0.4   4.0.0  3.1.3  3.0.5  3.0.1  2.1.4  2.1.0  2.0.1  1.10.5  1.10.1  1.8.6  1.8.2  1.7.4  1.7    1.6.2  1.5.4  1.5    1.4.2  1.3.3  1.2.9  1.2.5  1.2.1  1.1.3  1.0.2
  4.0.3   3.1.6  3.1.2  3.0.4  3.0.0  2.1.3  2.0.4  2.0.0  1.10.4  1.10.0  1.8.5  1.8.1  1.7.3  1.6.5  1.6.1  1.5.3  1.4.5  1.4.1  1.3.2  1.2.8  1.2.4  1.2    1.1.2  1.0.1
==> Remote versions (not yet checksummed):
==> Warning: Found no unchecksummed versions for openmpi
```



#### `spack arch`

通过执行`spack arch`命令来了解当前机器的CPU架构

```bash
$ spack arch                                                                                                                                                                                                    
linux-ubuntu18.04-cascadelake

```

通过使用`spack arch --known-targets`命令来显示所有CPU架构

??? "--known-targets"
    ```bash
    $ spack arch --known-target                                                                                                                                                                                     
    Generic architectures (families)
        aarch64  arm  ppc  ppc64  ppc64le  ppcle  sparc  sparc64  x86  x86_64

    GenuineIntel - x86
        i686  pentium2  pentium3  pentium4  prescott
    
    GenuineIntel - x86_64
        nocona  core2  nehalem  westmere  sandybridge  ivybridge  haswell  broadwell  skylake  mic_knl  skylake_avx512  cannonlake  cascadelake  icelake
    
    AuthenticAMD - x86_64
        k10  bulldozer  zen  piledriver  zen2  steamroller  excavator
    
    IBM - ppc64
        power7  power8  power9
    
    IBM - ppc64le
        power8le  power9le
    
    Cavium - aarch64
        thunderx2
    
    Fujitsu - aarch64
        a64fx
    
    ARM - aarch64
        graviton  graviton2
    
    ```



### 软件包参数指定



在安装时您可以使用：

-  `@`来指定需要安装的版本；

- `%`来选择需要使用的编译器；

- `+`或`-`或`~`来指定此软件的布尔值变量依赖（通过`spack info <package>`）来查找；

- `name=<value>`来指定此此软件的非布尔值环境依赖;

- `target=<value> os=<value>`指定CPU架构和系统（`target=skylake_avx512 os=linux`）

- `^`指定依赖的软件(`^callpath@1.1`)

  

### 安装和卸载软件



#### `spack install`

`spack install`可以从`spack list`中找到的软件中来安装软件包。

我们以`gcc`为例，

1.如果`gcc`依赖其他的软件包，spack将会先将依赖包安装，

2.然后校验下载的包文件没有错误，构建并安装到目录`SPACK_ROOT/opt`

在安装完成后，您可以看到成功安装的输出。`arch=skylake_avx512`为指定平台，可以参考`spack arch`。

在执行过程中出现的`[+]`为已经安装完成的依赖项，无须重新编译。

```bash
$ spack install gcc@10.2.0 target=skylake_avx512
[+] /home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/libsigsegv-2.12-65cvzrqkbruz3jwexeg7r44ovcz6zml5
[+] /home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/berkeley-db-18.1.40-6zahi2waipcqbqepcscqyc52alhv6bxw
[+] /home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/pkgconf-1.7.3-idmzwmyvgrql5cvpdof7tldya6jldl2g
[+] /home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/autoconf-archive-2019.01.06-zpx5jjtsx5qwkdoytknoqncsyaja7ryf
......
==> Installing gcc
==> No binary for gcc found: installing from source
==> Using cached archive: /home/user/spack/var/spack/cache/_source-cache/archive/b8/b8dd4368bb9c7f0b98188317ee0254dd8cc99d1e3a18d0ff146c855fe16c1d8c.tar.xz
==> gcc: Executing phase: 'autoreconf'
==> gcc: Executing phase: 'configure'
==> gcc: Executing phase: 'build'
==> gcc: Executing phase: 'install'

$spack find gcc                                                                                                                                                                                                
==> 2 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0  gcc@10.2.0

```



#### `spack uninstall`

卸载安装的软件包，您可以使用`spack uninstall <package>`来卸载目前已经安装的软件。

已经安装的软件可以通过`spack find`找到。

`--dependents`可以将依赖此软件的其他软件一起卸载。

如果您需要卸载所有软件，可以使用`--all`来实现

```bash
[~] spack uninstall --dependents openmpi%gcc@10.2.0
==> The following packages will be uninstalled:

    -- linux-ubuntu18.04-cascadelake / gcc@10.2.0 -------------------
    w2atmjq hdf5@1.10.7  sehyo5r netcdf-c@4.7.4  a4w6sg6 openmpi@3.1.6

==> Do you want to proceed? [y/N] y
==> Successfully uninstalled netcdf-c@4.7.4%gcc@10.2.0~dap~hdf4~jna+mpi~parallel-netcdf+pic+shared arch=linux-ubuntu18.04-cascadelake/sehyo5r
==> Successfully uninstalled hdf5@1.10.7%gcc@10.2.0~cxx~debug~fortran+hl~java+mpi+pic+shared~szip~threadsafe api=none arch=linux-ubuntu18.04-cascadelake/w2atmjq
==> Successfully uninstalled openmpi@3.1.6%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions+gpfs~java~legacylaunchers~lustre~memchecker~pmi~singularity~sqlite3+static~thread_multiple+vt+wrapper-rpath fabrics=none sche
dulers=none arch=linux-ubuntu18.04-cascadelake/a4w6sg6

```



#### `spack gc`

gc全称为（"Garbage Collector"），当您完成编译后，剩余一些目前不被其他软件依赖的项目可以被清理。

您可以通过使用`spack gc`来清理掉不需要的软件包。

```bash
$ spack gc                                                                                                                                                                                                       
==> The following packages will be uninstalled:                                                                                                                                                                    
                                                                                                                                                                                                                   
    -- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------                                                                                                                                              
    ayxupwc autoconf@2.69                nnarbli curl@7.72.0    wmcteof git@2.29.0            7vlbp2m libidn2@2.3.0        bhtzjl5 libxml2@2.9.10  66vt3mk perl@5.30.3         te3i3a5 xz@5.2.5                    
    zpx5jjt autoconf-archive@2019.01.06  zorqtw7 diffutils@3.7  h6rui4l hwloc@2.2.0           vhckpte libpciaccess@0.16    lj57qtx m4@1.4.18       idmzwmy pkgconf@1.7.3                                           
    oa4lssz automake@1.16.2              gqtxtgk expat@2.2.9    bez7ort libbsd@0.10.0         65cvzrq libsigsegv@2.12      phtjrqd openssh@8.4p1   mzpdmwj tar@1.32                                                
    6zahi2w berkeley-db@18.1.40          sz3iurw gdbm@1.18.1    upr5h3c libedit@3.1-20191231  smnwhui libtool@2.4.6        fcemob6 openssl@1.1.1h  nwu2trb texinfo@6.5                                             
    oukbgll bzip2@1.0.8                  4bstie7 gettext@0.21   qzdhx3t libiconv@1.16         2khexru libunistring@0.9.10  24pllrg pcre2@10.35     ozhfnrm util-macros@1.19.1                                      
                                                                                                                                                                                                                   
==> Do you want to proceed? [y/N] y                                                                                                                                                                                
==> Successfully uninstalled pkgconf@1.7.3%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/idmzwmy                                                                                                                 
==> Successfully uninstalled libtool@2.4.6%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/smnwhui                                                                                                                 
==> Successfully uninstalled autoconf@2.69%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/ayxupwc                                                                                                                 
==> Successfully uninstalled automake@1.16.2%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/oa4lssz                                                                                                               
==> Successfully uninstalled autoconf-archive@2019.01.06%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/zpx5jjt
==> Successfully uninstalled diffutils@3.7%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/zorqtw7
==> Successfully uninstalled git@2.29.0%gcc@7.5.0~tcltk arch=linux-ubuntu18.04-skylake_avx512/wmcteof
==> Successfully uninstalled util-macros@1.19.1%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512/ozhfnrm
==> Successfully uninstalled hwloc@2.2.0%gcc@7.5.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml+pci+shared arch=linux-ubuntu18.04-skylake_avx512/h6rui4l
......

$ spack find
==> 19 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0   gmp@6.1.2  isl@0.21  lua@5.3.5                  lua-luaposix@33.4.0  mpc@1.1.0   mpfr@4.0.2   readline@8.0  unzip@6.0    zstd@1.4.5
gcc@10.2.0  isl@0.18   lmod@8.3  lua-luafilesystem@1_7_0_2  mpc@1.1.0            mpfr@3.1.6  ncurses@6.2  tcl@8.6.10    zlib@1.2.11
```



#### 编译无法下载的软件包

Spack有一些软件包不能自动下载，原因可能有以下情况：

- 软件包作者需要用户在下载安装包前手动同意License（如jdk和galahd）
- 软件包无法在互联网上下载

如果需要编译这些软件包，您可能需要用到`spack mirrors`

创建Mirror步骤如下：

1.为mirror创建一个目录 ，您可以在任何目录创建，如`~/spackmirrors`

```bash
$ mkdir ~/spackmirrors
```

2.在`mirrors.yaml`文件中注册此路径,您可以通过`spack config edit mirrors`打开此文件

```bash
mirrors:
  manual: file://~/spackmirrors
```

3.将软件的二进制或源码压缩包放到此目录下，以`<package>/<package>-<version>.tar.gz`来命名。

例如，Oracle下载安装包需要手动同意License，我们放置Oracle的jdk的tar包到mirrors目录下：

```bash
$ cd ~/spackmirrors/jdk/
$ mv jdk-14_linux-x64_bin.tar.gz jdk-14.tar.gz 
$ tree spackmirrors                                                                                                                                                                                             
spackmirrors
└── jdk
    └── jdk-14.tar.gz
    
   
```

4.通过使用`<package>@<version>`来安装软件

```bash
$ spack install jdk@14 target=skylake_avx512
==> Warning: Missing a source id for jdk@14
==> Installing jdk
==> No binary for jdk found: installing from source
==> Warning: There is no checksum on file to fetch jdk@14 safely.
==>   Fetch anyway? [y/N] y
==> Fetching file:///home/user/spackmirrors/jdk/jdk-14.tar.gz
############################################################################################################################################################################################################ 100.0%
==> jdk: Executing phase: 'install'
[+] /home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/jdk-14-ctoqgkgu4qvmbbjvdewpdsx4hb4qawki
$ spack find jdk                                                                                                                                                                                              
==> 1 installed package
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
jdk@14

```



### 查询已安装的软件包

当您将软件完成安装时，您需要了解哪些软件是已经安装的，您可以通过下列命令来了解。

#### `spack find`

通过使用`spack find`可以找到已经安装的软件。

安装的软件包由名称、版本、编译器、架构和构建选项来分组。

```bash
$ spack find                                                                                                                 
==> 20 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0   gmp@6.1.2  isl@0.21  lmod@8.3   lua-luafilesystem@1_7_0_2  mpc@1.1.0  mpfr@3.1.6  ncurses@6.2   tcl@8.6.10  zlib@1.2.11
gcc@10.2.0  isl@0.18   jdk@14    lua@5.3.5  lua-luaposix@33.4.0        mpc@1.1.0  mpfr@4.0.2  readline@8.0  unzip@6.0   zstd@1.4.5

```

#### 查询`find`更多信息
通过拓展查找选项可以了解安装软件的详细信息

##### `spack find --deps`
通过使用`spack find --deps`您可以查找安装软件包所需要的依赖条件。

```bash
$ [~] spack find --deps gcc                                           
==> 2 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0
    gmp@6.1.2
    isl@0.18
    mpc@1.1.0
        mpfr@3.1.6
    zlib@1.2.11

gcc@10.2.0
    gmp@6.1.2
    isl@0.21
    mpc@1.1.0
        mpfr@4.0.2
    zlib@1.2.11
    zstd@1.4.5

```

现在我们可以看到`gcc`不同版本依赖软件的版本不同。

##### `spack find --paths`

通过使用`spack find --paths`您可以了解软件安装的位置。

```bash
[~] spack find --paths gcc@10.2.0                                                                                                           
==> 1 installed package
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@10.2.0  /home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gcc-10.2.0-7s57lgiv24lajskfc4aqf2npajnnweao

```



### 使用已安装的软件包

当您在系统中安装完成软件包，软件包会使用hash值来命名一个长路径。如果您想使用这个软件包，您可以使用`spack load`命令导入当前窗口的环境变量中。

#### `spack load`

当您执行安装操作后，通过`spack find`可以找到需要的软件包。

例如，如果我们需要导入`gcc`环境来使用，我们可以这样做

```bash
$ gcc --version                                                           
gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

                                                                     
$ spack find gcc                                                          
==> 2 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0  gcc@10.2.0

$ spack load gcc@10.2.0                                                   

$ gcc --version                                                           
gcc (Spack GCC) 10.2.0
Copyright (C) 2020 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```

当使用`spack load`命令后，spack将会将此软件的环境变量加载到环境变量`PATH`、`MANPATH`、`CPATH`和`LD_LIBRARY_PATH`中。

??? "用户环境变量"
    ```bash
    $ env|egrep "^PATH|^MANPATH|^CPATH|^LD_LIBRARY_PATH"                                                                                                                                                            
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/user/spack/bin
    MANPATH=/home/nfs/admin0/apps/lmod/lmod/share/man:
    ```

    $ spack load gcc@10.2.0 
    
    $ env|egrep "^PATH|^MANPATH|^CPATH|^LD_LIBRARY_PATH"                                                                                                                                                            
    PATH=/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gcc-10.2.0-7s57lgiv24lajskfc4aqf2npajnnweao/bin:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zstd-1.4.5-o4hfhwlx4awnsvlps3buomdnr7qprj2n/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/user/spack/bin
    MANPATH=/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gcc-10.2.0-7s57lgiv24lajskfc4aqf2npajnnweao/share/man:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zstd-1.4.5-o4hfhwlx4awnsvlps3buomdnr7qprj2n/share/man:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zlib-1.2.11-vk2i2rkkp2rz74b2g3gw3m27iupfeqt5/share/man:/home/nfs/admin0/apps/lmod/lmod/share/man:
    LD_LIBRARY_PATH=/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gcc-10.2.0-7s57lgiv24lajskfc4aqf2npajnnweao/lib64:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gcc-10.2.0-7s57lgiv24lajskfc4aqf2npajnnweao/lib:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zstd-1.4.5-o4hfhwlx4awnsvlps3buomdnr7qprj2n/lib:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zlib-1.2.11-vk2i2rkkp2rz74b2g3gw3m27iupfeqt5/lib:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/mpc-1.1.0-ldn2sunyabks46ttirwhmnztuoenvdtz/lib:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/mpfr-4.0.2-fveqzlfx3mgkzhjkjwkyvjrtipna5b5z/lib:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/isl-0.21-em2kdn4wsizcwret4hkevbzdiuftv746/lib:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gmp-6.1.2-s6jvux4ib2tfxvfhet7ven7yvtsmxbvb/lib
    CPATH=/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gcc-10.2.0-7s57lgiv24lajskfc4aqf2npajnnweao/include:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zstd-1.4.5-o4hfhwlx4awnsvlps3buomdnr7qprj2n/include:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/zlib-1.2.11-vk2i2rkkp2rz74b2g3gw3m27iupfeqt5/include:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/mpc-1.1.0-ldn2sunyabks46ttirwhmnztuoenvdtz/include:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/mpfr-4.0.2-fveqzlfx3mgkzhjkjwkyvjrtipna5b5z/include:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/isl-0.21-em2kdn4wsizcwret4hkevbzdiuftv746/include:/home/user/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-7.5.0/gmp-6.1.2-s6jvux4ib2tfxvfhet7ven7yvtsmxbvb/include


    ```

我们来做一个实验，用新安装的`gcc@10.2.0`来编译一个2048小游戏

项目地址： https://github.com/plibither8/2048.cpp

```bash
$ spack find gcc
==> 2 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0  gcc@10.2.0

$ spack load gcc@10.2.0
$ git clone https://github.com/plibither8/2048.cpp
$ cd 2048.cpp
$ mkdir build && cd build

$ cmake ../         
......
-- Build files have been written to: /home/nfs/admin0/qizewen/2048.cpp/build

$ cmake --build .        
Scanning dependencies of target 2048
......
[100%] Linking CXX executable 2048
[100%] Built target 2048

$ ./2048
```





#### `spack unload`

当您不希望在某个窗口中使用load后的软件，您可以通过`spack unload`来将环境变量已加载的环境变量取消。

```bash
[~] spack unload gcc                                                       
==> Warning: Reversing `Set` environment operation may lose original value                                                                                                                                         
==> Warning: Reversing `Set` environment operation may lose original value
```



#### 多版本共存问题

当您在load环境时可能会出现以下情况,您可以参考 **软件包参数指定** 章节指定更多参数来明确需要加载的软件包。

```bash
$ spack load gcc                                    
==> Error: gcc matches multiple packages.
  Matching packages:
    o7522oj gcc@7.5.0%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512
    7s57lgi gcc@10.2.0%gcc@7.5.0 arch=linux-ubuntu18.04-skylake_avx512
  Use a more specific spec.

$ spack load gcc@7.5.0 target=skylake_avx512
```



## Spack ENV环境

假设我们在做一个项目，但是在`spack find`列出的软件有多个版本，且不能迅速找到所需要的软件依赖。

那看起来是不是特别不方便？

这里Spack引入了ENV环境来创建一个干净的环境，您可以理解为类似Conda或Python Environment，您可以使用项目名称来命名ENV环境。

### `spack env create`

您可以使用`spack env create <envname>`创建虚拟环境。

```bash
$ spack env create emenv
                                                                   
==> Updating view at /home/user/spack/var/spack/environments/emenv/.spack-env/view
==> Created environment 'emenv' in /home/user/spack/var/spack/environments/emenv
==> You can activate this environment with:
==>   spack env activate emenv

```

!!! "info"
​	所有的命名的环境存储位置在`spack/var/spack/environments`中

虚拟环境最大的好处就是可以带走环境，您可以通过`spack/var/spack/environments`目录下的`spack.yaml`或`spack.lock`来创建新的虚拟环境。

以下示例通过一个已经创建的`spack.yaml`来创建新的虚拟环境`newenv`

```bash
$ pwd                                                                                       
/home/user/spack/var/spack/environments/emenv

$ ls                                                                                        
spack.lock  spack.yaml

$ spack env create newenv spack.yaml                                                        
==> Updating view at /home/user/spack/var/spack/environments/newenv/.spack-env/view
==> Created environment 'newenv' in /home/user/spack/var/spack/environments/newenv
==> You can activate this environment with:
==>   spack env activate newenv

```

您也可以通过使用`spack.lock`创建环境

```bash
$ spack env create myenv spack.lock
```



### `spack env activate <envname>`

在您创建完成虚拟环境后，您可以通过使用`spack env activate <envname> `;

在新环境中，您可以看到一个干净的env环境，那么您就可以只安装和此项目相关的软件包了。

```bash
$ spack env activate newenv         
$ spack find                        
                                                                                             
==> In environment emenv
==> No root specs
==> 0 installed packages

```



### `spack env deactive`

在使用完成虚拟环境时，您可以使用`spack env deactive`来取消使用虚拟环境。

```bash
$ spack find            
==> In environment emenv
==> No root specs
==> 0 installed packages

$ spack env deactivate                                                                                                                                                                         
$ spack find                                                                                                                                                                                   
==> 15 installed packages
-- linux-ubuntu18.04-skylake_avx512 / gcc@7.5.0 -----------------
gcc@7.5.0  gmp@6.1.2  isl@0.18  jdk@14  lmod@8.3  lua@5.3.5  lua-luafilesystem@1_7_0_2  lua-luaposix@33.4.0  mpc@1.1.0  mpfr@3.1.6  ncurses@6.2  readline@8.0  tcl@8.6.10  unzip@6.0  zlib@1.2.11


```

您也可以使用`despacktivate`来取消环境，效果与`spack env deactive`相同。

```bash
$ despacktivate
```





