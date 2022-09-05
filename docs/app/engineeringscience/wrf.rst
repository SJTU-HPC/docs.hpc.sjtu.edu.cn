.. _wrf:

WRF
====

简介
----

WRF(Weather Research and Forecasting Model)模式是有美国环境预报中心(NCEP),
美国国家大气研究中心（NCAR）以及多个大学、研究所和业务部门联合研发的一种统一的中尺度天气预报模式。
WRF模式适用范围很广，从中小尺度到全球尺度的数值预报和模拟都有广泛的应用。

WPS是预处理WRF运行数据的工具。

可用版本
--------

+--------+---------+----------+---------------------------------------------+
| 版本   | 平台    | 构建方式 | 模块名                                      |
+========+=========+==========+=============================================+
| 4.2.1  | |cpu|   | 源码     | wrf/4.2.1-oneapi-2021.4.0 思源一号          |
+--------+---------+----------+---------------------------------------------+
| 4.2    | |cpu|   | 源码     | wps/4.2-oneapi-2021.4.0 思源一号            |
+--------+---------+----------+---------------------------------------------+
| 4.3.1  | |cpu|   | 源码     | wrf_cmaq/5.3.3-wrf-4.3.1                    |
+--------+---------+----------+---------------------------------------------+

算例位置 
---------

.. code:: bash

   思源一号 : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/wrf-4.2/wrf_data
   π2.0    : /lustre/opt/contribute/cascadelake/wrf_cmaq/wrf_data
   
算例目录

.. code:: bash
               
   [hpc@node234 wrf-4.2]$ tree wrf_data/
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

思源一号和π2.0两个集群上的数据均是模拟2016年10月06日00点至2016年10月08日0点的气象数据
   
geog_data_path的位置
--------------------

.. code:: bash

   思源一号 : /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/geo/geog
   π2.0    : /lustre/opt/contribute/cascadelake/wrf_cmaq/geo

集群上的WRF和WPS
-----------------------

- `思源一号上的WRF和WPS`_
- `π2.0上的WRF和WPS`_

.. _思源一号上的WRF和WPS:

思源一号上的WRF和WPS
---------------------
   
自定义编译WRF和WPS
~~~~~~~~~~~~~~~~~~~

思源一号上已部署所依赖的库及版本
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

   hdf5-1.12.0             
   libpng-1.6.37 
   netcdf-c-4.8.1
   netcdf-fortran-4.5.3         
   zlib-1.2.11 
   jasper-1.900.29


自定义编译WRF
>>>>>>>>>>>>>

编译WRF前导入所需的环境变量

.. code:: bash

   module load oneapi
   export CC=icc
   export FC=ifort
   export F90=ifort
   export CXX=icpc
   export DIR=/dssg/home/acct-hpc/hpchgc/software/wrf/WRF_4.1.1_Intel/library
   export LD_LIBRARY_PATH=$DIR/wrf_libs_intel/lib:$LD_LIBRARY_PATH
   export LDFLAGS=-L$DIR/wrf_libs_intel/lib
   export CPPFLAGS=-I$DIR/wrf_libs_intel/include
   export NETCDF=$DIR/wrf_libs_intel/
   export HDF5=$DIR/wrf_libs_intel/
   export NETCDF=$DIR/wrf_libs_intel/
   export HDF5=$DIR/wrf_libs_intel/

.. code:: bash

   tar xvf v4.2.1.tar.gz
   cd WRF-4.2.1/
   ./configure 
   
可根据所需选择相应的参数，思源一号上的预编译版本选择的是20号，使用intel编译器编译WRF，并可以多节点并行运行。

.. code:: bash

   Please select from among the following Linux x86_64 options:

     1. (serial)   2. (smpar)   3. (dmpar)   4. (dm+sm)   PGI (pgf90/gcc)
     5. (serial)   6. (smpar)   7. (dmpar)   8. (dm+sm)   PGI (pgf90/pgcc): SGI MPT
     9. (serial)  10. (smpar)  11. (dmpar)  12. (dm+sm)   PGI (pgf90/gcc): PGI accelerator
    13. (serial)  14. (smpar)  15. (dmpar)  16. (dm+sm)   INTEL (ifort/icc)
                                            17. (dm+sm)   INTEL (ifort/icc): Xeon Phi (MIC architecture)
    18. (serial)  19. (smpar)  20. (dmpar)  21. (dm+sm)   INTEL (ifort/icc): Xeon (SNB with AVX mods)
    22. (serial)  23. (smpar)  24. (dmpar)  25. (dm+sm)   INTEL (ifort/icc): SGI MPT
    26. (serial)  27. (smpar)  28. (dmpar)  29. (dm+sm)   INTEL (ifort/icc): IBM POE
    30. (serial)               31. (dmpar)                PATHSCALE (pathf90/pathcc)
    32. (serial)  33. (smpar)  34. (dmpar)  35. (dm+sm)   GNU (gfortran/gcc)
    36. (serial)  37. (smpar)  38. (dmpar)  39. (dm+sm)   IBM (xlf90_r/cc_r)
    40. (serial)  41. (smpar)  42. (dmpar)  43. (dm+sm)   PGI (ftn/gcc): Cray XC CLE
    44. (serial)  45. (smpar)  46. (dmpar)  47. (dm+sm)   CRAY CCE (ftn $(NOOMP)/cc): Cray XE and XC
    48. (serial)  49. (smpar)  50. (dmpar)  51. (dm+sm)   INTEL (ftn/icc): Cray XC
    52. (serial)  53. (smpar)  54. (dmpar)  55. (dm+sm)   PGI (pgf90/pgcc)
    56. (serial)  57. (smpar)  58. (dmpar)  59. (dm+sm)   PGI (pgf90/gcc): -f90=pgf90
    60. (serial)  61. (smpar)  62. (dmpar)  63. (dm+sm)   PGI (pgf90/pgcc): -f90=pgf90
    64. (serial)  65. (smpar)  66. (dmpar)  67. (dm+sm)   INTEL (ifort/icc): HSW/BDW
    68. (serial)  69. (smpar)  70. (dmpar)  71. (dm+sm)   INTEL (ifort/icc): KNL MIC
    72. (serial)  73. (smpar)  74. (dmpar)  75. (dm+sm)   FUJITSU (frtpx/fccpx): FX10/FX100 SPARC64 IXfx/Xlfx

   Enter selection [1-75] : 

根据个人所需可选择mpi进行编译，思源一号部署的预编译版本的更改参数如下：

.. code:: bash

   更改文件configure.wrf的参数

   DM_FC           =       mpiifort
   DM_CC           =       mpiicc
   
自定义编译WPS
>>>>>>>>>>>>>>>
   
导入如下环境变量

.. code:: bash
               
   export WRF_DIR=../WRF-4.2.1/
   export JASPERLIB=$DIR/wrf_libs_intel/lib/
   export JASPERINC=$DIR/wrf_libs_intel/include/
    
    
.. code:: bash

   tar xvf v4.2.tar.gz
   cd WPS-4.2/
   ./configure
   
根据个人所需选择所需版本，思源一号上部署的预编译版本选择的19号，可多节点并行运行。（一般情况下选择17串行版即可满足计算所需）

.. code:: bash

   Please select from among the following supported platforms.

      1.  Linux x86_64, gfortran    (serial)
      2.  Linux x86_64, gfortran    (serial_NO_GRIB2)
      3.  Linux x86_64, gfortran    (dmpar)
      4.  Linux x86_64, gfortran    (dmpar_NO_GRIB2)
      5.  Linux x86_64, PGI compiler   (serial)
      6.  Linux x86_64, PGI compiler   (serial_NO_GRIB2)
      7.  Linux x86_64, PGI compiler   (dmpar)
      8.  Linux x86_64, PGI compiler   (dmpar_NO_GRIB2)
      9.  Linux x86_64, PGI compiler, SGI MPT   (serial)
     10.  Linux x86_64, PGI compiler, SGI MPT   (serial_NO_GRIB2)
     11.  Linux x86_64, PGI compiler, SGI MPT   (dmpar)
     12.  Linux x86_64, PGI compiler, SGI MPT   (dmpar_NO_GRIB2)
     13.  Linux x86_64, IA64 and Opteron    (serial)
     14.  Linux x86_64, IA64 and Opteron    (serial_NO_GRIB2)
     15.  Linux x86_64, IA64 and Opteron    (dmpar)
     16.  Linux x86_64, IA64 and Opteron    (dmpar_NO_GRIB2)
     17.  Linux x86_64, Intel compiler    (serial)
     18.  Linux x86_64, Intel compiler    (serial_NO_GRIB2)
     19.  Linux x86_64, Intel compiler    (dmpar)
     20.  Linux x86_64, Intel compiler    (dmpar_NO_GRIB2)
     21.  Linux x86_64, Intel compiler, SGI MPT    (serial)
     22.  Linux x86_64, Intel compiler, SGI MPT    (serial_NO_GRIB2)
     23.  Linux x86_64, Intel compiler, SGI MPT    (dmpar)
     24.  Linux x86_64, Intel compiler, SGI MPT    (dmpar_NO_GRIB2)
     25.  Linux x86_64, Intel compiler, IBM POE    (serial)
     26.  Linux x86_64, Intel compiler, IBM POE    (serial_NO_GRIB2)
     27.  Linux x86_64, Intel compiler, IBM POE    (dmpar)
     28.  Linux x86_64, Intel compiler, IBM POE    (dmpar_NO_GRIB2)
     29.  Linux x86_64 g95 compiler     (serial)
     30.  Linux x86_64 g95 compiler     (serial_NO_GRIB2)
     31.  Linux x86_64 g95 compiler     (dmpar)
     32.  Linux x86_64 g95 compiler     (dmpar_NO_GRIB2)
     33.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (serial)
     34.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (serial_NO_GRIB2)
     35.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (dmpar)
     36.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (dmpar_NO_GRIB2)
     37.  Cray XC CLE/Linux x86_64, Intel compiler   (serial)
     38.  Cray XC CLE/Linux x86_64, Intel compiler   (serial_NO_GRIB2)
     39.  Cray XC CLE/Linux x86_64, Intel compiler   (dmpar)
     40.  Cray XC CLE/Linux x86_64, Intel compiler   (dmpar_NO_GRIB2)
   
   Enter selection [1-40] :



思源一号上使用预编译的WRF和WPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先用WPS处理数据
>>>>>>>>>>>>>>>>>

1. 由于WPS处理数据需要复杂的文件依赖关系，可先拷贝WPS目录中的文件到本地

.. code:: bash

   mkdir ~/data && cd ~/data
   mkdir WRF && cd WRF
   cp -r /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/wrf-4.2/WPS-4.2 ./
 
2. 拷贝数据到WPS目录中进行数据处理

.. code:: bash

   cd WPS-4.2
   cp -r /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/wrf-4.2/wrf_data/* ./
   
3. namelist.wps文件内容设置如下：

.. code:: bash

   &share
   wrf_core = 'ARW',
   max_dom = 1,
   start_date = '2016-10-06_00:00:00'
   end_date   = '2016-10-08_00:00:00'
   interval_seconds = 21600
   io_form_geogrid = 2,
  /

  &geogrid
   parent_id         =   1,
   parent_grid_ratio =   1,
   i_parent_start    =   1,
   j_parent_start    =   1,
   e_we              =  515,
   e_sn              =  515,
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! The default datasets used to produce the MAXSNOALB and ALBEDO12M
   ! fields have changed in WPS v4.0. These fields are now interpolated
   ! from MODIS-based datasets.
   !
   ! To match the output given by the default namelist.wps in WPS v3.9.1,
   ! the following setting for geog_data_res may be used:
   !
   ! geog_data_res = 'maxsnowalb_ncep+albedo_ncep+default',     'maxsnowalb_ncep+albedo_ncep+default', 
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   geog_data_res = 'default','default',
   dx = 12000,
   dy = 12000,
   map_proj = 'lambert',
   ref_lat   =  31.00,
   ref_lon   = 120.00,
   ref_x = 351
   ref_y = 208
   truelat1  =  30.0,
   truelat2  =  60.0,
   stand_lon = 120.0,
   geog_data_path = '/dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/geo/geog/'
  /

  &ungrib
   out_format = 'WPS',
   prefix = 'FILE',
  /

  &metgrid
   fg_name = 'FILE'
   io_form_metgrid = 2, 
  /
  
4. 运行geogrid.exe程序定义模型投影、区域范围，嵌套关系，对地表参数进行插值。

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load oneapi
   module load wps
   geogrid.exe 
   
5.根据模拟时期选择文件

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load oneapi
   module load wps
   link_grib.csh fnl_2016100*
   cp ungrib/Variable_Tables/Vtable.GFS Vtable

6.从grib数据中提取所需要的气象参数

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load oneapi
   module load wps
   ungrib.exe 
   
7.将气象参数插值到模拟区域

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load oneapi
   module load wps
   metgrid.exe 

WRF运行
>>>>>>>>

1. 由于WRF运行数据需要复杂的文件依赖关系，可先拷贝WRF目录中必要的文件到本地

.. code:: bash

   cd ~/data
   cd WRF
   mkdir WRF-4.2.1 && cd WRF-4.2.1
   cp -r /dssg/opt/icelake/linux-centos8-icelake/oneapi-2021.4.0/wrf_cmaq/wrf-4.2/WRF-4.2.1/run/* ./

2. 拷贝WPS生成的met文件到WRF-4.2.1目录

.. code:: bash

   cp -r ../WPS-4.2/met_em.d01.2016-10-0* ./
   
3. namelist.input文件内容设置如下，参数需要与wps的namelist.wps参数一致：

.. code:: bash

    &time_control
    run_days                            = 2,
    run_hours                           = 0,
    run_minutes                         = 0,
    run_seconds                         = 0,
    start_year                          = 2016,
    start_month                         = 10,
    start_day                           = 06,
    start_hour                          = 00,
    end_year                            = 2016,
    end_month                           = 10,
    end_day                             = 08,
    end_hour                            = 00,
    interval_seconds                    = 21600
    input_from_file                     = .true.,.true.,
    history_interval                    = 60,   60,
    frames_per_outfile                  = 12,   12,
    restart                             = .false.,
    restart_interval                    = 5000,
    io_form_history                     = 2
    io_form_restart                     = 2
    io_form_input                       = 2
    io_form_boundary                    = 2
    /

    &domains
    time_step                           = 60,
    time_step_fract_num                 = 0,
    time_step_fract_den                 = 1,
    max_dom                             = 1,
    e_we                                = 515,    112,
    e_sn                                = 515,    97,
    e_vert                              = 33,    33,
    p_top_requested                     = 5000,
    num_metgrid_levels                  = 32,
    num_metgrid_soil_levels             = 4,
    dx                                  = 12000,
    dy                                  = 12000,
    grid_id                             = 1,     2,
    parent_id                           = 0,     1,
    i_parent_start                      = 1,     31,
    j_parent_start                      = 1,     17,
    parent_grid_ratio                   = 1,     3,
    parent_time_step_ratio              = 1,     3,
    feedback                            = 1,
    smooth_option                       = 0
    /

    &physics
    physics_suite                       = 'tropical'
    mp_physics                          = 6,    -1,
    cu_physics                          = 16,    -1,
    ra_lw_physics                       = 4,    -1,
    ra_sw_physics                       = 4,    -1,
    bl_pbl_physics                      = 8,    8,
    sf_sfclay_physics                   = 1,    1,
    sf_surface_physics                  = 2,    -1,
    radt                                = 12,    30,
    bldt                                = 0,     0,
    cudt                                = 5,     5,
    icloud                              = 1,
    num_land_cat                        = 21,
    sf_urban_physics                    = 0,     0,     0,
    /

    &fdda
    /

    &dynamics
    hybrid_opt                          = 2, 
    w_damping                           = 0,
    diff_opt                            = 1,      1,
    km_opt                              = 4,      4,
    diff_6th_opt                        = 0,      0,
    diff_6th_factor                     = 0.12,   0.12,
    base_temp                           = 290.
    damp_opt                            = 3,
    zdamp                               = 5000.,  5000.,
    dampcoef                            = 0.2,    0.2,
    khdif                               = 0,      0,
    kvdif                               = 0,      0,
    non_hydrostatic                     = .true., .true.,
    moist_adv_opt                       = 1,      1,     
    scalar_adv_opt                      = 1,      1,     
    gwd_opt                             = 0,      1,
    /

    &bdy_control
    spec_bdy_width                      = 5,
    specified                           = .true.
    /

    &grib2
    /

    &namelist_quilt
    nio_tasks_per_group = 0,
    nio_groups = 1,
    /
   

4. 运行real.exe程序，脚本如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g 
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   
   module load oneapi
   module load wrf
   ulimit -s unlimited
   real.exe
  
5. 运行wrf.exe程序，脚本如下，该部分是最终也是最耗时的执行程序。

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=64c512g 
   #SBATCH -N 4
   #SBATCH --ntasks-per-node=64
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load oneapi
   module load wrf
   ulimit -s unlimited
   mpirun wrf.exe

.. _π2.0上的WRF和WPS:

π2.0上的WRF和WPS
--------------------

π2.0上先用WPS处理数据 
~~~~~~~~~~~~~~~~~~~~~~

1. 由于WPS处理数据需要复杂的文件依赖关系，可先拷贝WPS目录中的文件到本地

.. code:: bash

   mkdir ~/data && cd ~/data
   mkdir WRF && cd WRF
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/packet_1/WPS-4.3.1 ./
 
2. 拷贝数据到WPS目录中进行数据处理

.. code:: bash

   cd WPS-4.3.1
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/wrf_data/* ./
   
3. namelist.wps文件内容设置如下：

.. code:: bash

   &share
   wrf_core = 'ARW',
   max_dom = 1,
   start_date = '2016-10-06_00:00:00'
   end_date   = '2016-10-08_00:00:00'
   interval_seconds = 21600
   io_form_geogrid = 2,
  /

  &geogrid
   parent_id         =   1,
   parent_grid_ratio =   1,
   i_parent_start    =   1,
   j_parent_start    =   1,
   e_we              =  515,
   e_sn              =  515,
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! The default datasets used to produce the MAXSNOALB and ALBEDO12M
   ! fields have changed in WPS v4.0. These fields are now interpolated
   ! from MODIS-based datasets.
   !
   ! To match the output given by the default namelist.wps in WPS v3.9.1,
   ! the following setting for geog_data_res may be used:
   !
   ! geog_data_res = 'maxsnowalb_ncep+albedo_ncep+default',     'maxsnowalb_ncep+albedo_ncep+default', 
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   geog_data_res = 'default','default',
   dx = 12000,
   dy = 12000,
   map_proj = 'lambert',
   ref_lat   =  31.00,
   ref_lon   = 120.00,
   ref_x = 351
   ref_y = 208
   truelat1  =  30.0,
   truelat2  =  60.0,
   stand_lon = 120.0,
   geog_data_path = '/lustre/opt/contribute/cascadelake/wrf_cmaq/geo/'
  /

  &ungrib
   out_format = 'WPS',
   prefix = 'FILE',
  /

  &metgrid
   fg_name = 'FILE'
   io_form_metgrid = 2, 
  /
  
4. 运行geogrid.exe程序定义模型投影、区域范围，嵌套关系，对地表参数进行插值。

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load wrf_cmaq/5.3.3-wrf-4.3.1
   
   geogrid.exe 
   
5.根据模拟时期选择文件

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load wrf_cmaq/5.3.3-wrf-4.3.1
   
   link_grib.csh fnl_2016100*
   cp ungrib/Variable_Tables/Vtable.GFS Vtable

6.从grib数据中提取所需要的气象参数

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load wrf_cmaq/5.3.3-wrf-4.3.1
   
   ungrib.exe 
   
7.将气象参数插值到模拟区域

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load wrf_cmaq/5.3.3-wrf-4.3.1
   
   metgrid.exe 

π2.0上运行WRF
~~~~~~~~~~~~~~~~

1. 由于WRF运行数据需要复杂的文件依赖关系，可先拷贝WRF目录中必要的文件到本地

.. code:: bash

   cd ~/data
   cd WRF
   mkdir WRF-4.3.1 && cd WRF-4.3.1
   cp -r /lustre/opt/contribute/cascadelake/wrf_cmaq/packet_1/WRF-master/run/* ./

2. 拷贝WPS生成的met文件到WRF-4.3.1目录

.. code:: bash

   cp -r  ~/data/WRF/WPS-4.3.1/met_em.d*  ./
   
3. namelist.input文件内容设置如下，参数需要与wps的namelist.wps参数一致：

.. code:: bash

    &time_control
    run_days                            = 2,
    run_hours                           = 0,
    run_minutes                         = 0,
    run_seconds                         = 0,
    start_year                          = 2016,
    start_month                         = 10,
    start_day                           = 06,
    start_hour                          = 00,
    end_year                            = 2016,
    end_month                           = 10,
    end_day                             = 08,
    end_hour                            = 00,
    interval_seconds                    = 21600
    input_from_file                     = .true.,.true.,
    history_interval                    = 60,   60,
    frames_per_outfile                  = 12,   12,
    restart                             = .false.,
    restart_interval                    = 5000,
    io_form_history                     = 2
    io_form_restart                     = 2
    io_form_input                       = 2
    io_form_boundary                    = 2
    /

    &domains
    time_step                           = 60,
    time_step_fract_num                 = 0,
    time_step_fract_den                 = 1,
    max_dom                             = 1,
    e_we                                = 515,    112,
    e_sn                                = 515,    97,
    e_vert                              = 33,    33,
    p_top_requested                     = 5000,
    num_metgrid_levels                  = 32,
    num_metgrid_soil_levels             = 4,
    dx                                  = 12000,
    dy                                  = 12000,
    grid_id                             = 1,     2,
    parent_id                           = 0,     1,
    i_parent_start                      = 1,     31,
    j_parent_start                      = 1,     17,
    parent_grid_ratio                   = 1,     3,
    parent_time_step_ratio              = 1,     3,
    feedback                            = 1,
    smooth_option                       = 0
    /

    &physics
    physics_suite                       = 'tropical'
    mp_physics                          = 6,    -1,
    cu_physics                          = 16,    -1,
    ra_lw_physics                       = 4,    -1,
    ra_sw_physics                       = 4,    -1,
    bl_pbl_physics                      = 8,    8,
    sf_sfclay_physics                   = 1,    1,
    sf_surface_physics                  = 2,    -1,
    radt                                = 12,    30,
    bldt                                = 0,     0,
    cudt                                = 5,     5,
    icloud                              = 1,
    num_land_cat                        = 21,
    sf_urban_physics                    = 0,     0,     0,
    /

    &fdda
    /

    &dynamics
    hybrid_opt                          = 2, 
    w_damping                           = 0,
    diff_opt                            = 1,      1,
    km_opt                              = 4,      4,
    diff_6th_opt                        = 0,      0,
    diff_6th_factor                     = 0.12,   0.12,
    base_temp                           = 290.
    damp_opt                            = 3,
    zdamp                               = 5000.,  5000.,
    dampcoef                            = 0.2,    0.2,
    khdif                               = 0,      0,
    kvdif                               = 0,      0,
    non_hydrostatic                     = .true., .true.,
    moist_adv_opt                       = 1,      1,     
    scalar_adv_opt                      = 1,      1,     
    gwd_opt                             = 0,      1,
    /

    &bdy_control
    spec_bdy_width                      = 5,
    specified                           = .true.
    /

    &grib2
    /

    &namelist_quilt
    nio_tasks_per_group = 0,
    nio_groups = 1,
    /
   

4. 运行real.exe程序，脚本如下：

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load wrf_cmaq/5.3.3-wrf-4.3.1
   ulimit -s unlimited
   real.exe
  
5. 运行wrf.exe程序，脚本如下，该部分是最终也是最耗时的执行程序。

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=test
   #SBATCH --partition=cpu
   #SBATCH -N 1
   #SBATCH --ntasks-per-node=40
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   module load wrf_cmaq/5.3.3-wrf-4.3.1
   ulimit -s unlimited
   mpirun wrf.exe

运行结果(单位为：秒，越低越好)
------------------------------

思源一号上WRF的运行时间
~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------------------+
|              wrf/4.2.1-oneapi-2021.4.0         |
+=============+==========+===========+===========+
| 核数        | 64       | 128       | 256       |
+-------------+----------+-----------+-----------+
| Exec time   | 0:36:21  | 0:18:05   | 0:10:44   |
+-------------+----------+-----------+-----------+

π2.0上WRF的运行时间
~~~~~~~~~~~~~~~~~~~~

+------------------------------------------------+
|           wrf_cmaq/5.3.3-wrf-4.3.1             |
+=============+==========+===========+===========+
| 核数        | 40       | 80        | 160       |
+-------------+----------+-----------+-----------+
| Exec time   | 1:10:28  | 0:42:22   | 0:26:01   |
+-------------+----------+-----------+-----------+

参考资料
--------

-  `WRF 官网 <https://www.mmm.ucar.edu/weather-research-and-forecasting-model>`__
