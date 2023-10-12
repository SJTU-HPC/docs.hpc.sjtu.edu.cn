如何使用KOS队列
=================
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

登录kos登陆节点
~~~~~~~~~~~~~~~~~

-  从Pi集群登陆节点跳转

::

   ssh username@koslogin1

-  使用ssh软件工具登录

::

   host: koslogin.hpc.sjtu.edu.cn

申请KOS队列
~~~~~~~~~~~~~

| kos队列名称为：kos
| 下面介绍两种方式如何申请kos队列，kos队列使用方法与原CPU队列基本一致，只需要将队列名称从cpu改为kos

slurm脚本

::

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=kos
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=1

   hostname

srun交互式作业

::

   srun -n 1 -p kos --pty /bin/bash

已部署应用列表
----------------

可以通过module avail查询kos队列上已部署的编译器，应用等

modulefile路径

::

   /lustre/share/spack/modules/cascadelake/linux-rhel8-skylake_avx512

::

   --------------------------------------------- /lustre/share/spack/modules/cascadelake/linux-rhel8-skylake_avx512 ---------------------------------------------
      boost/1.82.0-gcc-8.5.0     (D)    intel-oneapi-compilers/2021.4.0-gcc-8.5.0        netcdf-fortran/4.5.2-intel-2021.4.0
      bowtie/1.3.1-gcc-8.5.0     (D)    intel-oneapi-mkl/2021.4.0-gcc-8.5.0              netcdf-fortran/4.6.0-gcc-8.5.0          (D)
      bwa/0.7.17-gcc-8.5.0              intel-oneapi-mkl/2023.1.0-gcc-8.5.0       (D)    openblas/0.3.23-gcc-8.5.0               (D)
      cgal/4.13-gcc-8.5.0               intel-oneapi-mpi/2021.4.0-gcc-8.5.0              openfoam/2206-gcc-8.5.0                 (D)
      cmake/3.26.3-gcc-8.5.0     (D)    jasper/2.0.32-gcc-8.5.0                          openmpi/4.1.5-gcc-8.5.0                 (D)
      eigen/3.4.0-gcc-8.5.0             jasper/2.0.32-intel-2021.4.0                     perl/5.36.0-gcc-8.5.0
      fftw/3.3.8-intel-2021.4.0         jasper/3.0.3-gcc-8.5.0                    (D)    perl/5.36.0-intel-2021.4.0              (D)
      fftw/3.3.10-gcc-8.5.0      (D)    lammps/20220623.3-gcc-8.5.0                      python/3.10.10-gcc-8.5.0                (D)
      git/2.40.0-gcc-8.5.0       (D)    lammps/20220623.3-intel-2021.4.0                 quantum-espresso/6.7-gcc-8.5.0-openblas
      gromacs/2022.5-gcc-8.5.0   (D)    llvm/12.0.1-gcc-8.5.0                            quantum-espresso/6.7-gcc-8.5.0
      hdf5/1.10.6-gcc-8.5.0             miniconda3/22.11.1                        (D)    samtools/1.16.1-gcc-8.5.0               (D)
      hdf5/1.10.6-intel-2021.4.0        mpich/4.1.1-gcc-8.5.0                     (D)    stream/5.10-gcc-8.5.0
      hdf5/1.14.1-2-gcc-8.5.0    (D)    netcdf-c/4.9.2-gcc-8.5.0                         wps/4.4-intel-2021.4.0
      hpl/2.3-gcc-8.5.0                 netcdf-c/4.9.2-intel-2021.4.0             (D)    wrf/4.4.2-gcc-8.5.0
      hypre/2.28.0-gcc-8.5.0     (D)    netcdf-fortran/4.5.2-gcc-8.5.0                   wrf/4.4.2-intel-2021.4.0                (D)

测试结果对比
--------------

根据多个常用的科学应用测试结果来看，计算速度上KOS系统与CentOS7系统基本保持一致。

+-----------+---------------+----------------+
|app        | KOS           |CentOS          |
+===========+===============+================+
|hpl        | 3651.1 Gflops |3774.26 Gflops  |
+-----------+---------------+----------------+
| gromacs   | 18.558 ns/day | 17.266 ns/day  |
+-----------+---------------+----------------+ 
|lammps     | 0:03:13       | 0:03:16        |  
+-----------+---------------+----------------+
| cp2k      | 46.039        | 45.721         |
+-----------+---------------+----------------+
|relion     | 00:28:45      | 00:25:00       |
+-----------+---------------+----------------+
| qe        |10m18.08s      | 9m21.27s       |
+-----------+---------------+----------------+
| openfoam  | 00:03:16      | 00:03:28       | 
+-----------+---------------+----------------+
| wrf       | 01:15:24      | 1:10:28        | 
+-----------+---------------+----------------+
| bwa       | index: 1909   | index: 1989.35 | 
+-----------+---------------+----------------+

应用使用文档
--------------

下面以gromacs应用使用为例介绍如何使用KOS队列的计算节点进行计算，使用方法与原CPU队列基本保持一致，只需要更改队列名称为kos并调用正确的module名称即可。

gromacs
~~~~~~~~

版本：2022.5

::

   module load gromacs/2022.5-gcc-8.5.0

功能测试

准备算例

::

   wget -c https://ftp.gromacs.org/pub/benchmarks/water_GMX50_bare.tar.gz --no-check-certificate
   tar xf water_GMX50_bare.tar.gz
   cd water-cut1.0_GMX50_bare/0768/

准备作业脚本

::

   #!/bin/bash
   #SBATCH --job-name=gromacs      # 作业名
   #SBATCH --partition=kos
   #SBATCH -N 2
   #SBATCH --ntasks-per-node=40      # 每节点核数
   #SBATCH --exclusive
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gromacs/2022.5-gcc-8.5.0
   mpirun -np 1 gmx_mpi grompp -f pme.mdp
   mpirun -np $SLURM_NPROCS gmx_mpi mdrun -dlb yes -v -nsteps 10000 -resethway -noconfout -pin on -ntomp 1 -s topol.tpr

测试结果

::


                  Core t (s)   Wall t (s)        (%)
          Time:     3724.895       46.567     7999.1
                    (ns/day)    (hour/ns)
   Performance:       18.558        1.293

编译应用
----------

因为系统版本升级，用户原有编译软件需要重新编译才能使用，编译方式和原先系统流程基本一致，根据需要使用的编译器调用对应模块即可，下面以fftw为例，介绍如何在kos系统上使用gcc和intel两种编译器编译软件。
### 先申请计算节点用于编译

::

   srun -n 1 -p kos --pty /bin/bash

gcc+openmpi
~~~~~~~~~~~~~

::

   module load openmpi/4.1.5-gcc-8.5.0

   wget https://fftw.org/pub/fftw/fftw-3.3.8.tar.gz
   tar -xvf fftw-3.3.8.tar.gz
   cd fftw-3.3.8/
   ./configure --prefix=$PWD --enable-mpi --enable-openmp --enable-threads --enable-shared MPICC=mpicc CC=gcc F77=gfortran
   make
   make install

intel-oneapi
~~~~~~~~~~~~~~

::

   module load intel-oneapi-compilers/2021.4.0-gcc-8.5.0
   module load intel-oneapi-mpi/2021.4.0-gcc-8.5.0

   wget https://fftw.org/pub/fftw/fftw-3.3.8.tar.gz
   tar -xvf fftw-3.3.8.tar.gz
   cd fftw-3.3.8/
   ./configure --prefix=$PWD --enable-mpi --enable-openmp --enable-threads --enable-shared MPICC=mpiicc CC=icc F77=ifort
   make
   make install
