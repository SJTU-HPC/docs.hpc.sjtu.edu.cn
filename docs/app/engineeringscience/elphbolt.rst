.. _elphbolt:

Elphbolt
========

简介
----
Elphbolt用于计算物理领域中的电子输运，采用的是严格电声耦合结合玻尔兹曼输运的方法。

Elphbolt的基本使用
-------------------------------

1. 准备两个目录来存放相关参数文件：

.. code:: bash

  

   workdir="./Si_6r4_300K_CBM_gcc/"
   inputdir="./input"

   mkdir $workdir
   mkdir $inputdir


2. 在input目录下编写如下input.nml文件：

.. code:: bash

   &allocations
        numelements=1
        numatoms=2
  /

  &crystal_info
	name = 'Cubic Si'
	elements="Si"
        atomtypes=1 1
	lattvecs(:,1) =  -0.27010011   0.00000000   0.27010011
        lattvecs(:,2) =   0.00000000   0.27010011   0.27010011
        lattvecs(:,3) =  -0.27010011   0.27010011   0.00000000   
        basis(:,1) =    0.00 0.00 0.00
        basis(:,2) =    0.25 0.25 0.25
        T = 300.0 !K
        epsilon0 = 11.7 !From Ioffe
  /

  &electrons
	spindeg = 2
	indlowband = 5 !Lowest transport band
	indhighband = 6 !Highest transport band
	indlowconduction = 5 !Lowest conduction band
	numbands = 8 !Total wannier bands
	enref = 6.70035 !eV, CBM
	chempot = 6.70035 !eV, CBM
  /

  &numerics
	qmesh = 6 6 6
	mesh_ref = 4 !kmesh = 24 24 24
	fsthick = 0.4 !eV about enref
	datadumpdir = './scratch/' !Or, enter suitable scratch directory
        conv_thres = 0.0001
        maxiter = 50 !Maximum number of iterations
  /

  &wannier
    coarse_qmesh = 6 6 6
  /

3. 在Si_6r4_300K_CBM_gcc目录下执行以下命令：

.. code:: bash

  cp ../$inputdir/input.nml .

  ln -s ../$inputdir/rcells_g .
  ln -s ../$inputdir/rcells_k .
  ln -s ../$inputdir/rcells_q .
  ln -s ../$inputdir/wsdeg_g .
  ln -s ../$inputdir/wsdeg_k .
  ln -s ../$inputdir/wsdeg_q .
  ln -s ../$inputdir/epwdata.fmt .
  ln -s ../$inputdir/epmatwp1 .
  ln -s ../$inputdir/FORCE_CONSTANTS_3RD .
  ln -s ../$inputdir/espresso.ifc2 .

4. 在Si_6r4_300K_CBM_gcc目录下编写以下elphbolt.slurm脚本：

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=elphbolt       
   #SBATCH --partition=small
   #SBATCH -N 1           
   #SBATCH --ntasks-per-node=4
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err

   module load gcc/8.3.0
   module load openmpi/4.0.4-gcc-8.3.0
   module load netlib-lapack/3.8.0-gcc-8.3.0
   module load openblas/0.3.7-gcc-8.3.0
   module load elphbolt/1.0.0-gcc-8.3.0-openmpi-4.0.4
   
   cafrun -n 2 elphbolt.x


5. 使用如下指令提交作业：

.. code:: bash

   sbatch elphbolt.slurm

6. 作业完成后可在.out文件中看到如下结果(部分)：

.. code:: bash

  +-------------------------------------------------------------------------+
  | \                                                                       |
  |  \                                                                      |
  |   \   \                                                                 |
  |    \   \                                                                |
  |   __\   \              _        _    _           _    _                 |
  |   \      \         ___|.|      |.|  | |__   ___ |.|_ / /__              |
  |    \    __\       / _ \.|   _  |.|_ | '_ \ / _ \|.|_  ___/              |
  |     \  \         |  __/.| |/ \_|/  \| |_) : (_) |.|/ /__                |
  |      \ \          \___|_|/|__/ |   /| ___/ \___/|_|\___/                |
  |       \ \                /|                                             |
  |        \\                \|                                             |
  |         \\                '                                             |
  |          \                                                              |
  |           \                                                             |
  | A solver for the coupled electron-phonon Boltzmann transport equations. |
  | Copyright (C) 2020- Nakib Haider Protik.                                |
  |                                                                         |
  | This is a 'free as in freedom'[*] software, distributed under the GPLv3.|
  | [*] https://www.gnu.org/philosophy/free-sw.en.html                      |
  +-------------------------------------------------------------------------+
  
  Number of coarray images =     2
  ___________________________________________________________________________
  ______________________________________________________Setting up crystal...
  Material: Cubic Si                                                                                            
  Isotopic average of masses will be used.
  Si mass =   0.28085510E+02 u
  Lattice vectors [nm]:
  -0.27010011E+00   0.00000000E+00   0.27010011E+00
  0.00000000E+00   0.27010011E+00   0.27010011E+00
  -0.27010011E+00   0.27010011E+00   0.00000000E+00
  Primitive cell volume =  0.39409804E-01 nm^3
  Reciprocal lattice vectors [1/nm]:
  -0.11631216E+02  -0.11631216E+02   0.11631216E+02
  0.11631216E+02   0.11631216E+02   0.11631216E+02
  -0.11631216E+02   0.11631216E+02  -0.11631216E+02
  Brillouin zone volume =  0.15943204E+03 1/nm^3
  Crystal temperature =  300.00 K
  ___________________________________________________________________________
  ____________________________________________Reading numerics information...
  q-mesh =     6    6    6
  k-mesh =    24   24   24
  Fermi window thickness (each side of reference energy) =   0.40000000E+00 eV
  Working directory = /lustre/home/acct-hpc/hpcpzz/Elphbolttest/Si_6r4_300K_CBM_gcc
  Data dump directory = ./scratch/
  T-dependent data dump directory = ./scratch/T0.300E+03
  e-ph directory = ./scratch/g2
  ph-ph directory = ./scratch/V2
  Reuse e-ph matrix elements: F
  Reuse ph-e matrix elements: F
  Reuse ph-ph matrix elements: F
  Reuse ph-ph transition probabilities: F
  Use tetrahedron method: F
  Include ph-e interaction: T
  Include ph-isotope interaction: F
  Include ph-substitution interaction: F
  Include electron-charged impurity interaction: F
  Include drag: T
  Plot quantities along path: F
  Maximum number of BTE iterations =    50
  BTE convergence threshold =   0.10000000E-03
  ___________________________________________________________________________
  ______________________________________________________Analyzing symmetry...
  Crystal symmetry group = Fd-3m
  Number of crystal symmetries (without time-reversal) =    48
  ___________________________________________________________________________
  _________________________________________Reading EPW Wannier information...

