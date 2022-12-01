.. _flash:

FLASH
======

简介
----
FLASH是一款开源的适用于等离子物理和天体物理的磁流体动力学（MHD）模拟软件，它是由罗切斯特大学物理与天文学系Flash计算科学中心开发。其包含流体动力学（Unsplit PPM, WENO, PCM, GP; Split PPM; 2T+Radiation）、磁流体动力学（Unsplit Staggered Mesh; Split 8 wave）、扩展磁动力学（Magnetic Resistivity, Hall, Biermann Battery）、状态方程（理想气体, Degenerate电离等离子体）、辐射输运（Multigroup Flux-limited Diffusion）等物理求解器

Flash软件依赖项
-----------------

- 支持Fortran 90的编译器
- MPICH
- HDF5串行版本
- PnetCDF
- Chombo
- HYPRE
- ...


pi2.0上自定义编译FLASH
----------------------

- 软件下载。FLASH软件下载需要在官网（https://flash.rochester.edu/site/flashcode）填写申请，获批后才能下载。

- 申请计算节点并加载模块。（本示例使用的依赖项包括：平台已编译的mpich和hdf5、需自编译的PnetCDF）

.. code:: bash

    srun -p small -n 4 --pty /bin/bash
    module load mpich/3.4.2-gcc-9.2.0 hdf5/1.10.6-gcc-9.2.0

- 编译并安装PnetCDF。Flash输出文件使用并行的NetCDF格式，因此需额外编译

.. code:: bash

    git clone https://github.com/Parallel-NetCDF/PnetCDF.git
    cd PnetCDF
    autoreconf -i
    ./configure --prefix=(path_to_installation_pnetcdf) ## 用户可以自行指定pnetcdf的安装路径，此处使用(path_to_installation_pnetcdf)替代
    make -j
    make install

- 配置Flash。本文档以配置Flash，生成2D Sedov爆炸问题源代码为例

.. code:: bash

    tar -xvf FLASHX.Y.tar ## 解压下载的Flash软件
    cd FLASHX.Y
    ./setup Sedov -auto ## 使用setup脚本配置Flash

- 修改编译文件。配置完成后源目录下将生成object文件夹，其中包含配置Flash求解2D Sedov爆炸问题所需的流体动力学求解器、状态方程、网格单元等模块。但其中的编译文件Makefile.h为默认值，需将其中的MPI_PATH、HDF5_PATH和NCMPI_PATH变量进行修改（为便于展示，本示例只使用mpich、hdf5和pnetcdf三个依赖项）。

.. code:: bash

    cd object
    vi Makefile.h ## 手动修改文件中的MPI_PATH、HDF5_PATH和NCMPI_PATH变量，如NCMPI_PATH=(path_to_installation_pnetcdf)，MPI_PATH和HDF5_PATH的路径可以使用env命令查看
    make -j

- 编译结束。编译完成后将在object目录下生成名称为flash4的可执行文件

- 新建目录并编写以下Flash参数文件flash.par。参数文件使用官网提供示例（https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node13.html）

.. code:: bash

    # runtime parameters
    basenm  = "sedov_"
    lrefine_max = 5
    refine_var_1 = "dens"
    refine_var_2 = "pres"
    restart = .false.
    checkpointFileIntervalTime = 0.01
    nend = 10000
    tmax = 0.05
    gamma = 1.4
    xl_boundary_type = "outflow"
    xr_boundary_type = "outflow"
    yl_boundary_type = "outflow"
    yr_boundary_type = "outflow"
    plot_var_1 = "dens"
    plot_var_2 = "temp"
    plot_var_3 = "pres"
    sim_profFileName = "/dev/null"

- 编写脚本。在此目录下编写以下flash.slurm脚本，其中(path_to_installation_flash)为flash安装路径，

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --partition=cpu
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=40
    #SBATCH --time=1-00:00:00
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    module purge
    module load mpich/3.4.2-gcc-9.2.0 hdf5/1.10.6-gcc-9.2.0

    ulimit -s unlimited
    ulimit -l unlimited

    mpirun -np $SLURM_NTASKS (path_to_installation_flash)/object/flash4 -par_file flash.par

- 使用 ``sbatch`` 提交作业：

.. code:: bash

   sbatch flash.slurm

- 编译结束后在当前目录下生成如下文件：

.. code:: bash

    ./
    ├── amr_runtime_parameters.dump
    ├── flash.par
    ├── flash.slurm
    ├── LargestSummary.out
    ├── sedov.dat
    ├── sedov_forced_hdf5_plt_cnt_0000
    ├── sedov_hdf5_chk_0000
    ├── sedov_hdf5_chk_0001
    ├── sedov_hdf5_chk_0002
    ├── sedov_hdf5_chk_0003
    ├── sedov_hdf5_chk_0004
    ├── sedov_hdf5_chk_0005
    ├── sedov_hdf5_plt_cnt_0000
    └── sedov.log
