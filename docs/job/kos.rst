KOS队列使用
============
KOS系统简介
-------------

Pi集群cpu队列目前采用的是CentOS
7操作系统，该系统版本将于2024年6月底停止维护，这将使得Pi集群超算系统面临安全隐患，为应对
CentOS
7即将全面停服带来的安全风险，我们计划将Pi集群计算节点的操作系统从CentOS
7替换为浪潮信息的国产KOS系统。

KOS系统是浪潮信息基于Linux
Kernel、OpenAnolis等开源技术自主研发的一款服务器操作系统，支持x86、ARM等主流架构处理器，性能和稳定性居于行业领先地位，具备成熟的
CentOS
迁移和替换能力，可满足云计算、大数据、分布式存储、人工智能、边缘计算等应用场景需求。

下文将介绍如何在Pi集群上使用KOS系统节点，已部署的应用列表，应用测试结果对比，应用的使用文档，以及如何自行编译应用。

如何使用KOS系统队列
---------------------

登录KOS登陆节点
~~~~~~~~~~~~~~~~~
-  使用ssh软件工具登录

::

   host: koslogin.hpc.sjtu.edu.cn

申请KOS队列
~~~~~~~~~~~~~~~~~

| kos队列名称为：kos
| kos队列使用方法与原CPU队列基本一致，只需要将作业脚本或srun命令中的队列名称从cpu改为kos

slurm脚本
^^^^^^^^^^

::

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=kos
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1

   hostname

srun交互式作业
^^^^^^^^^^^^^^

::

   srun -n 1 -p kos --pty /bin/bash

已部署应用列表
---------------

可以通过module avail查询kos队列上已部署的编译器，应用等

modulefile路径

::

   /lustre/share/spack/modules/cascadelake/linux-rhel8-skylake_avx512

::

   --------------------------------- /lustre/share/spack/modules/cascadelake/linux-rhel8-skylake_avx512 ---------------------------------
   boost/1.82.0-gcc-8.5.0                   intel-oneapi-compilers/2021.4.0            netcdf-fortran/4.6.0-gcc-8.5.0          (D)
   bowtie/1.3.1-gcc-8.5.0                   intel-oneapi-mkl/2021.4.0                  openblas/0.3.23-gcc-8.5.0
   bwa/0.7.17-gcc-8.5.0                     intel-oneapi-mkl/2023.1.0           (D)    openfoam/2206-gcc-8.5.0                 (D)
   cgal/4.13-gcc-8.5.0                      intel-oneapi-mpi/2021.4.0                  openmpi/4.1.5-gcc-8.5.0
   cmake/3.26.3-gcc-8.5.0                   jasper/2.0.32-gcc-8.5.0                    perl/5.36.0-gcc-8.5.0
   cp2k/2023.2-intel-oneapi-2021.4.0        jasper/2.0.32-intel-2021.4.0               perl/5.36.0-intel-2021.4.0              (D)
   eigen/3.4.0-gcc-8.5.0                    jasper/3.0.3-gcc-8.5.0              (D)    python/3.10.10-gcc-8.5.0
   fftw/3.3.8-intel-2021.4.0                lammps/20220623.3-gcc-8.5.0                quantum-espresso/6.7-gcc-8.5.0-openblas
   fftw/3.3.10-gcc-8.5.0             (D)    lammps/20220623.3-intel-2021.4.0           quantum-espresso/6.7-gcc-8.5.0          (D)
   git/2.40.0-gcc-8.5.0                     llvm/12.0.1-gcc-8.5.0                      samtools/1.16.1-gcc-8.5.0
   gromacs/2022.5-gcc-8.5.0          (D)    miniconda3/22.11.1                         stream/5.10-gcc-8.5.0
   hdf5/1.10.6-gcc-8.5.0                    mpich/4.1.1-gcc-8.5.0                      wps/4.4-intel-2021.4.0
   hdf5/1.10.6-intel-2021.4.0               netcdf-c/4.9.2-gcc-8.5.0                   wrf/4.4.2-gcc-8.5.0
   hdf5/1.14.1-2-gcc-8.5.0           (D)    netcdf-c/4.9.2-intel-2021.4.0       (D)    wrf/4.4.2-intel-2021.4.0                (D)
   hpl/2.3-gcc-8.5.0                        netcdf-fortran/4.5.2-gcc-8.5.0
   hypre/2.28.0-gcc-8.5.0                   netcdf-fortran/4.5.2-intel-2021.4.0

测试结果对比
--------------

根据多个常用的科学应用测试结果来看，在计算速度上KOS系统与CentOS7系统基本保持一致。
其中hpl计算单位为Gflops，数值越大计算速度越快，gromacs计算单位为ns/day，数值越大计算速度越快，其余应用计算单位为计算时间（h:m:s），数值越小计算速度越快。

+-----------+---------------+----------------+
|app        | KOS           |CentOS          |
+===========+===============+================+
| hpl       | 3651.1 Gflops |3774.26 Gflops  |
+-----------+---------------+----------------+
| gromacs   | 18.558 ns/day | 17.266 ns/day  |
+-----------+---------------+----------------+ 
|lammps     | 00:03:13      | 00:03:16       |  
+-----------+---------------+----------------+
| cp2k      | 00:00:46      | 00:00:45       |
+-----------+---------------+----------------+
|relion     | 00:28:45      | 00:25:00       |
+-----------+---------------+----------------+
| qe        |00:10:08       | 00:09:21       |
+-----------+---------------+----------------+
| openfoam  | 00:03:16      | 00:03:28       | 
+-----------+---------------+----------------+
| wrf       | 01:15:24      | 01:10:28       | 
+-----------+---------------+----------------+
| bwa       | 00:31:49      | 00:33:09       | 
+-----------+---------------+----------------+

应用测试案例
-------------
下面介绍上述应用使用的计算算例，应用使用文档可点击应用名称查看原Pi2.0集群的CPU队列使用文档。注意需要将作业脚本中的队列名称改为kos，应用的调用模块需要改为kos队列中的模块名。

`hpl <https://docs.hpc.sjtu.edu.cn/app/benchtools/hpl.html#id4>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HPL.dat
::

   HPLinpack benchmark input file
   Innovative Computing Laboratory, University of Tennessee
   HPL.out      output file name (if any)
   7            device out (6=stdout,7=stderr,file)
   1            # of problems sizes (N)
   176640 Ns
   1            # of NBs
   256 NBs
   0            PMAP process mapping (0=Row-,1=Column-major)
   1            # of process grids (P x Q)
   8 Ps
   10 Qs
   16.0         threshold
   3            # of panel fact
   0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
   2            # of recursive stopping criterium
   2 4          NBMINs (>= 1)
   1            # of panels in recursion
   2            NDIVs
   3            # of recursive panel fact.
   0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
   1            # of broadcast
   0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
   1            # of lookahead depth
   0            DEPTHs (>=0)
   2            SWAP (0=bin-exch,1=long,2=mix)
   64           swapping threshold
   0            L1 in (0=transposed,1=no-transposed) form
   0            U  in (0=transposed,1=no-transposed) form
   1            Equilibration (0=no,1=yes)
   8            memory alignment in double (> 0)


`gromacs <https://docs.hpc.sjtu.edu.cn/app/engineeringscience/gromacs.html#>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gromacs选择的测试算例为gromacs提供的benchmark水分子算例，本文选取的为0768水分子算例。

获取算例：  
::

   wget -c https://ftp.gromacs.org/pub/benchmarks/water_GMX50_bare.tar.gz
   tar xf water_GMX50_bare.tar.gz
   cd water-cut1.0_GMX50_bare/0768/
   tree 0768/
   0768/
   ├── conf.gro
   ├── pme.mdp
   ├── rf.mdp
   └── topol.top

`lammps <https://docs.hpc.sjtu.edu.cn/app/engineeringscience/lammps.html#in-lj>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lammps选择的测试算例为lammps官方benchmark算例:in.lj  

输入文件内容
::

   # 3d Lennard-Jones melt

   variable     x index 4
   variable     y index 4
   variable     z index 4

   variable     xx equal 20*$x
   variable     yy equal 20*$y
   variable     zz equal 20*$z

   units                lj
   atom_style   atomic

   lattice              fcc 0.8442
   region               box block 0 ${xx} 0 ${yy} 0 ${zz}
   create_box   1 box
   create_atoms 1 box
   mass         1 1.0

   velocity     all create 1.44 87287 loop geom

   pair_style   lj/cut 2.5
   pair_coeff   1 1 1.0 1.0 2.5

   neighbor     0.3 bin
   neigh_modify delay 0 every 20 check no

   fix          1 all nve

`cp2k <https://docs.hpc.sjtu.edu.cn/app/engineeringscience/cp2k.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cp2k选择的测试算例为官方benchmark中的H2O-64.inp算例

算例获取：
:: 

   cp -rfv /lustre/opt/cascadelake/linux-rhel8-skylake_avx512/intel-2021.4.0/cp2k/cp2k/benchmarks/QS/H2O-64.inp .

`quantum-espresso <https://docs.hpc.sjtu.edu.cn/app/engineeringscience/quantum-espresso.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
quantum-espresso选择的测试算例为官方提供的test—cases中的small算例：ausurf.in

算例获取：
::

   wget https://repository.prace-ri.eu/git/UEABS/ueabs/-/raw/master/quantum_espresso/test_cases/small/ausurf.in
   wget https://repository.prace-ri.eu/git/UEABS/ueabs/-/raw/master/quantum_espresso/test_cases/small/Au.pbe-nd-van.UPF

`openfoam <https://docs.hpc.sjtu.edu.cn/app/engineeringscience/openfoam.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
openfoam测试算例选择的是simpleFoam求解器计算摩托车外流场motorBike算例

算例获取：
::

   module load openfoam/2206-gcc-8.5.0
   mkdir openfoamTest1
   cd openfoamTest1
   cp -rv $FOAM_TUTORIALS  ./
   cd ./tutorials/incompressible//simpleFoam/motorBike

`wrf <https://docs.hpc.sjtu.edu.cn/app/engineeringscience/wrf.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wrf测试算例选择的是模拟2016年10月06日00点至2016年10月08日0点的气象数据

算例获取：
::

   /lustre/opt/contribute/cascadelake/wrf_cmaq/wrf_data
   
   tree wrf_data/
   wrf_data/
   ├── fnl_20161006_00_00.grib2
   ├── fnl_20161006_06_00.grib2
   ├── fnl_20161006_12_00.grib2
   ├── fnl_20161006_18_00.grib2
   ├── fnl_20161007_00_00.grib2
   ├── fnl_20161007_06_00.grib2
   ├── fnl_20161007_12_00.grib2
   ├── fnl_20161007_18_00.grib2
   └── fnl_20161008_00_00.grib2

   geog_data_path:
   /lustre/opt/contribute/cascadelake/wrf_cmaq/geo

`bwa <https://docs.hpc.sjtu.edu.cn/app/bioinformatics/bwa.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bwa选择的测试算例是B17NC_R1.fastq

算例获取：

::

   mkdir ~/bwa && cd ~/bwa
   cp -r /lustre/share/sample/bwa/* ./
   gzip -d B17NC_R1.fastq.gz
   gzip -d B17NC_R2.fastq.gz

`relion <https://docs.hpc.sjtu.edu.cn/app/bioinformatics/relion.html#>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
relion选择的测试算例为人类去铁铁蛋白（apo­ferritin）电镜图像数据集，总计 32933 张图像，数据量 8.1 GB

算例获取：
::

   mkdir relion-test
   cd relion-test
   cp -rfv /lustre/share/samples/kos-samples/relion/apo-ferritin .


编译应用
---------

因为系统版本升级，用户原有编译软件需要重新编译才能使用，编译方式和原先系统流程基本一致，根据需要使用的编译器调用对应模块即可，下面以fftw为例，介绍如何在kos系统上使用gcc和intel两种编译器编译软件。
### 先申请计算节点用于编译

::

   srun -n 1 -p kos --pty /bin/bash

gcc+openmpi
~~~~~~~~~~~~

::

   module load openmpi/4.1.5-gcc-8.5.0

   wget https://fftw.org/pub/fftw/fftw-3.3.8.tar.gz
   tar -xvf fftw-3.3.8.tar.gz
   cd fftw-3.3.8/
   ./configure --prefix=$PWD --enable-mpi --enable-openmp --enable-threads --enable-shared MPICC=mpicc CC=gcc F77=gfortran
   make
   make install

intel-oneapi
~~~~~~~~~~~~~

::

   module load intel-oneapi-compilers/2021.4.0-gcc-8.5.0
   module load intel-oneapi-mpi/2021.4.0-gcc-8.5.0

   wget https://fftw.org/pub/fftw/fftw-3.3.8.tar.gz
   tar -xvf fftw-3.3.8.tar.gz
   cd fftw-3.3.8/
   ./configure --prefix=$PWD --enable-mpi --enable-openmp --enable-threads --enable-shared MPICC=mpiicc CC=icc F77=ifort
   make
   make install
