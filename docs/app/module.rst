软件模块使用方法
====================

module 命令
-------------

集群软件以 module 形式供全局调用。常见的 module 命令如下

``module load [MODULE]``: 加载模块

``module avail`` 或 ``module av`` : 列出所有模块

``module av intel``: 列出含有 intel 名字的所有模块

``module list``: 列出所有已加载的模块

``module show [MODULE]``: 列出该模块的信息，如路径、环境变量等

也可以一次加载或卸载多个模块。

.. code:: bash

   $ module load gcc openmpi
   $ module unload gcc openmpi

如果您喜欢最新的稳定版本，则可以忽略版本号（默认加载带有 D 标识的版本）。

下方两句命令效果一致：

.. code:: bash

   $ module load gcc openmpi
   $ module load gcc/9.3.0-gcc-4.8.5 openmpi/4.0.5-gcc-9.2.0

::

   ----------------------------------------------------- /lustre/share/spack/modules/cascadelake/linux-centos7-x86_64 -----------------------------------------------------
   abinit/8.10.3-gcc-9.2.0-openblas-openmpi        hdf5/1.10.6-gcc-9.2.0-openmpi                            netcdf-fortran/4.5.2-intel-19.0.4-impi   (D)
   alphafold/2-python-3.8                          hdf5/1.10.6-gcc-9.2.0                                    netlib-lapack/3.8.0-intel-19.0.4         (D)
   bcftools/1.9-gcc-9.2.0                   (D)    hdf5/1.10.6-intel-19.0.4-impi                            nwchem/6.8.1-intel-19.0.4-impi
   bedtools2/2.27.1-intel-19.0.4            (D)    hdf5/1.10.6-intel-19.0.5-impi                            openblas/0.3.7-gcc-9.2.0                 (D)
   bismark/0.19.0-intel-19.0.4                     hdf5/1.10.6-intel-19.0.5-openmpi                  (D)    openjdk/1.8.0_222-b10-gcc-9.2.0
   boost/1.70.0-gcc-9.2.0                          hisat2/2.1.0-intel-19.0.4                         (D)    openjdk/11.0.2-gcc-9.2.0
   boost/1.70.0-intel-19.0.4                       hypre/2.20.0-gcc-9.2.0-openblas-openmpi           (D)    openjdk/11.0.2-intel-19.0.4              (D)
   boost/1.70.0-intel-19.0.5                (D)    intel-mkl/2019.3.199-intel-19.0.4                        openmpi/3.1.5-gcc-9.2.0
   bowtie/1.2.3-gcc-9.2.0                   (D)    intel-mkl/2019.5.281-intel-19.0.5                        openmpi/3.1.5-gcc-9.3.0
   bowtie2/2.3.5.1-intel-19.0.4                    intel-mkl/2020.1.217-intel-19.1.1                 (D)    openmpi/3.1.5-intel-19.0.5
   bwa/0.7.17-gcc-9.2.0                            intel-mpi/2019.4.243-intel-19.0.4                        openmpi/4.0.5-gcc-9.2.0                  (D)
   bwa/0.7.17-intel-19.0.4                  (D)    intel-mpi/2019.6.154-gcc-9.2.0                           perl/5.30.0-gcc-9.2.0
   cdo/1.9.8-gcc-9.2.0                             intel-mpi/2019.6.154-intel-19.0.5                 (D)    perl/5.30.0-gcc-9.3.0
   cp2k/8.2-gcc-9.2.0-openblas              (D)    intel-parallel-studio/cluster.2019.4-intel-19.0.4        perl/5.30.0-intel-19.0.4
   cuda/9.0.176-intel-19.0.4                       intel-parallel-studio/cluster.2019.5-intel-19.0.5        perl/5.30.0-intel-19.0.5
   cuda/10.1.243-gcc-9.2.0                         intel-parallel-studio/cluster.2020.1-intel-19.1.1 (D)    perl/5.30.0-intel-19.1.1                 (D)
   cuda/10.2.89-intel-19.0.4                (D)    jasper/2.0.16-gcc-9.2.0                                  picard/2.19.0-gcc-9.2.0
   cufflinks/2.2.1-gcc-9.2.0                       jdk/12.0.2_10-gcc-9.2.0                                  picard/2.19.0-intel-19.0.4               (D)
   cufflinks/2.2.1-intel-19.0.4             (D)    jdk/12.0.2_10-intel-19.0.4                        (D)    python/2.7.16-intel-19.0.4
   eigen/3.3.7-gcc-9.2.0                    (D)    json-fortran/7.1.0-gcc-9.2.0                             python/2.7.16-intel-19.1.1
   fastqc/0.11.7-gcc-9.2.0                         lammps/20200721-intel-19.0.5-openmpi                     python/3.7.4-gcc-9.2.0
   fastqc/0.11.7-intel-19.0.4               (D)    lammps/20210310-intel-19.0.5-openmpi                     python/3.7.4-intel-19.0.4
   fftw/3.3.8-gcc-9.2.0-openmpi                    lammps/20210702-intel-19.0.5-impi                        python/3.7.4-intel-19.0.5
   fftw/3.3.8-gcc-9.2.0                            libxc/4.3.2-gcc-9.2.0                                    quantum-espresso/6.4.1-intel-19.0.4-impi
   fftw/3.3.8-gcc-9.3.0-openmpi                    libxc/4.3.2-intel-19.0.4                          (D)    quantum-espresso/6.4.1-intel-19.0.5-impi
   fftw/3.3.8-intel-19.0.4-impi                    lumpy-sv/0.2.13-gcc-9.2.0                                quantum-espresso/6.5-intel-19.0.4-impi
   fftw/3.3.8-intel-19.0.5-impi                    mcl/14-137-gcc-9.2.0                                     quantum-espresso/6.5-intel-19.0.5-impi
   fftw/3.3.8-intel-19.0.5-openmpi                 megahit/1.1.4-intel-19.0.4                        (D)    rna-seqc/1.1.8-gcc-9.2.0
   fftw/3.3.8-intel-19.1.1-impi                    metis/5.1.0-gcc-9.2.0                             (D)    rna-seqc/1.1.8-intel-19.0.4              (D)
   fftw/3.3.9-gcc-9.2.0-openmpi                    mpich/3.3.2-gcc-9.2.0                                    rosettafold/1-python-3.8
   fftw/3.3.9-gcc-9.3.0                     (D)    mpich/3.3.2-intel-19.0.4                                 samtools/1.9-gcc-9.2.0
   flash/1.2.11-gcc-9.2.0                   (D)    mpich/3.3.2-intel-19.0.5                          (D)    samtools/1.9-intel-19.0.4                (D)
   fsl/6.0-fsl-gcc-4.8.5                           mvapich2/2.3.2-intel-19.0.5                       (D)    siesta/4.0.1-intel-19.0.4-impi
   gatk/3.8-1-gcc-9.2.0                     (D)    namd/2.14-gcc-9.2.0-openmpi                              stream/5.10-intel-19.0.4
   gromacs/2019.2-gcc-9.2.0-openmpi                nano/4.7-gcc-4.8.5                                       stream/5.10-intel-19.0.5                 (D)
   gromacs/2019.4-gcc-9.2.0-openmpi                nektar/5.0.0-intel-19.0.4-impi                           sumo/1.10.0-sumo
   gromacs/2020.2-gcc-9.2.0-openmpi                netcdf-c/4.7.3-gcc-9.2.0-openmpi                         sundials/3.1.2-gcc-9.2.0
   gromacs/2021-gcc-9.3.0-openmpi                  netcdf-c/4.7.3-gcc-9.2.0                                 svaba/1.1.3-gcc-4.8.5
   gsl/2.5-gcc-9.2.0                               netcdf-c/4.7.3-intel-19.0.4-impi                         tophat/2.1.2-intel-19.0.4
   gsl/2.5-intel-19.0.4                            netcdf-c/4.7.3-intel-19.0.5-impi                         vsearch/2.4.3-intel-19.0.4               (D)
   gsl/2.5-intel-19.0.5                     (D)    netcdf-c/4.7.3-intel-19.0.5-openmpi               (D)    wrf/4.2-gcc-9.2.0-openmpi
   hdf5/1.10.5-intel-19.0.4-impi                   netcdf-fortran/4.5.2-gcc-9.2.0-openmpi


在SLURM上，我们应用了以下规则来选取最合适的模块。

1. 编译器：如果加载了\ ``gcc``\ 或\ ``icc``\ ，请根据相应的编译器加载已编译的模块。或在必要时加载默认的编译器\ ``gcc``\ 。
2. MPI库：如果已加载其中一个库（\ ``openmpi``\ ，\ ``impi``\ ，\ ``mvapich2``\ ，\ ``mpich``\ ），加载针对相应MPI编译的模块。在必要的时候,默认MPI
   lib中的\ ``openmpi``\ 将被装载。
3. Module版本：每个模块均有默认版本，如果未指定版本号，则将加载该默认版本。



参考资料
--------

- Lmod: A New Environment Module System https://lmod.readthedocs.io/en/latest/
- Environment Modules Project http://modules.sourceforge.net/
- Modules Software Environment on NERSC https://www.nersc.gov/users/software/nersc-user-environment/modules/
