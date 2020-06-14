# <center>应用软件</center>

-----------

本文档介绍 Pi 上可用的 module 软件模块，和一些常用的科学软件。<br><br>
Pi 上的 module 软件模块均可直接调用。<br>
未在 module 里列出的软件，需先安装后才能使用。常用的开源软件可以联系我们安装，商业软件则需用户自行获取版权并安装。


## Pi 上的 module 软件模块

最新列表可通过下方命令查看：
```
$ module av
```
然后通过 module load 加载需要的模块，示例：
```
$ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
```

| module | Version |
| ---- | ---- |
| abinit | 8.10.3-gcc-8.3.0-openblas-openmpi |
| bcftools | 1.9-gcc-9.2.0, 1.9-gcc-8.3.0 |
| bedtools2 | 2.27.1-intel-19.0.4, 2.27.1-gcc-8.3.0 |
| bismark | 0.19.0-intel-19.0.4 |
| boost | 1.70.0-gcc-9.2.0, 1.70.0-intel-19.0.4, 1.70.0-intel-19.0.5, 1.56.0-gcc-8.3.0, 1.65.1-gcc-8.3.0-openmpi, 1.66.0-gcc-8.3.0, 1.70.0-gcc-7.4.0, 1.70.0-gcc-8.3.0 |
| bowtie | 1.2.3-gcc-9.2.0, 1.2.3-gcc-8.3.0 |
| bwa | 0.7.17-gcc-9.2.0, 0.7.17-intel-19.0.4, 0.7.17-gcc-8.3.0 |
| cdo | 1.9.8-gcc-8.3.0-openmpi |
| cp2k | 6.1-gcc-8.3.0-openblas-openmpi |
| cuda | 10.2.89-intel-19.0.4, 10.1.243-gcc-9.2.0, 9.0.176-intel-19.0.4, 10.0.130-gcc-8.3.0, 10.1.243-gcc-8.3.0, 10.2.89-gcc-8.3.0, 9.0.176-gcc-8.3.0, 9.1.85-gcc-8.3.0, 9.2.88-gcc-8.3.0, 10.0.130-gcc-4.8.5, 10.1.243-gcc-4.8.5, 7.5.18-gcc-4.8.5, 8.0.61-gcc-4.8.5, 9.0.176-gcc-4.8.5, 9.2.88-gcc-4.8.5 |
| cudnn | 7.1.2-9.0-linux-x64-gcc-8.3.0, 7.6.5.32-10.1-linux-x64-gcc-4.8.5, 7.6.5.32-9.0-linux-x64-gcc-4.8.5 |
| cufflinks | 2.2.1-gcc-9.2.0, 2.2.1-intel-19.0.4, 2.2.1-gcc-8.3.0 |
| curl | 7.63.0-gcc-8.3.0 |
| eigen | 3.3.7-gcc-9.2.0, 3.3.7-gcc-8.3.0 |
| elpa | 2017.11.001-gcc-8.3.0-openblas-openmpi |
| emacs | 26.3-gcc-4.8.5 |
| fastqc | 0.11.7-gcc-9.2.0, 0.11.7-intel-19.0.4, 0.11.7-gcc-8.3.0 |
| fftw | 3.3.8-gcc-9.2.0, 3.3.8-intel-19.0.4, 3.3.8-intel-19.0.4-impi, 3.3.8-intel-19.0.5, 3.3.8-intel-19.0.5-impi, 2.1.5-gcc-8.3.0, 3.3.8-gcc-7.4.0, 3.3.8-gcc-8.3.0, 3.3.8-gcc-8.3.0-openmpi, 3.3.8-gcc-4.8.5 |
| flash | 1.2.11-gcc-9.2.0, 1.2.11-gcc-8.3.0 |
| gatk | 3.8-1-gcc-8.3.0 |
| gcc | 5.5.0-gcc-4.8.5, 7.4.0-gcc-4.8.5, 8.3.0-gcc-4.8.5, 9.2.0-gcc-4.8.5, 9.3.0-gcc-4.8.5 |
| gmap-gsnap | 2019-05-12-gcc-8.3.0 |
| graphmap | 0.3.0-gcc-8.3.0 |
| gromacs | 2019.2-gcc-9.2.0-openmpi, 2019.4-gcc-9.2.0-openmpi, 2019.4-intel-19.0.4-impi, 2019.2-gcc-8.3.0-openmpi, 2019.4-gcc-8.3.0-openmpi |
| gsl | 2.5-gcc-9.2.0, 2.5-intel-19.0.4, 2.5-intel-19.0.5, 2.5-gcc-8.3.0, 2.5-gcc-4.8.5 |
| hdf5 | 1.10.5-gcc-9.2.0, 1.10.5-intel-19.0.4-impi, 1.10.5-intel-19.0.5-impi, 1.10.6-intel-19.0.5-impi, 1.10.5-gcc-7.4.0, 1.10.6-gcc-8.3.0, 1.10.6-gcc-8.3.0-openmpi, 1.10.5-gcc-4.8.5
| hisat2 | 2.1.0-intel-19.0.4, 2.1.0-gcc-8.3.0
| intel | 18.0.4-gcc-4.8.5, 19.0.4-gcc-4.8.5, 19.0.5-gcc-4.8.5, 19.1.1-gcc-4.8.5
| intel-mkl | 2019.3.199-intel-19.0.4, 2019.5.281-intel-19.0.5, 2020.1.217-intel-19.1.1, 2019.4.243-intel-19.0.4, 2019.6.154-gcc-9.2.0, 2019.6.154-intel-19.0.5
| intel-parallel-studio | cluster.2019.4-intel-19.0.4, cluster.2019.5-intel-19.0.5, cluster.2020.1-intel-19.1.1, cluster.2018.4-intel-18.0.4
| jdk | 12.0.2_10-gcc-9.2.0, 12.0.2_10-intel-19.0.4, 12.0.2_10-gcc-8.3.0, 12.0.2_10-gcc-4.8.5
| lammps | 20190807-intel-19.0.5-impi, 20190807-gcc-8.3.0-openmpi
| libbeagle | 3.1.2-gcc-8.3.0
| libxc | 4.3.2-gcc-8.3.0
| llvm | 7.0.0-gcc-4.8.5
| lumpy-sv | 0.2.13-gcc-9.2.0
| mcl | 14-137-gcc-9.2.0
| megahit | 1.1.4-intel-19.0.4
| megahit | 1.1.4-gcc-8.3.0
| metis | 5.1.0-gcc-7.4.0, 5.1.0-gcc-8.3.0, 5.1.0-gcc-4.8.5
| miniconda2 | 4.6.14-gcc-4.8.5, 4.7.12.1-gcc-4.8.5, 4.6.14-gcc-4.8.5, 4.7.12.1-gcc-4.8.5, 4.8.2-gcc-4.8.5
| mpich | 3.3.2-gcc-9.2.0, 3.3.2-intel-19.0.4, 3.3.2-intel-19.0.5, 3.3.2-gcc-8.3.0
| mrbayes | 3.2.7a-gcc-8.3.0-openmpi
| mvapich2 | 2.3.2-intel-19.0.5, 2.3.2-gcc-8.3.0
| ncbi-rmblastn | 2.2.28-gcc-4.8.5
| nccl | 2.4.8-1-gcc-8.3.0
| netcdf-c | 4.7.3-gcc-9.2.0, 4.7.3-intel-19.0.5-impi, 4.7.3-gcc-7.4.0, 4.7.3-gcc-8.3.0, 4.7.3-gcc-8.3.0-openmpi
| netcdf-fortran | 4.5.2-gcc-9.2.0, 4.5.2-gcc-8.3.0, 4.5.2-gcc-8.3.0-openmpi
| nwchem | 6.8.1-gcc-9.2.0-openblas
| octave | 5.2.0-gcc-8.3.0-openblas
| openblas | 0.3.7-gcc-8.3.0, 0.3.7-gcc-9.2.0, 0.3.7-intel-19.0.4
| openfoam | 1712-gcc-7.4.0-openmpi, 1812_191001-gcc-7.4.0-openmpi, 1912-gcc-7.4.0-openmpi
| openfoam-org | 7-gcc-7.4.0-openmpi
| openjdk | 11.0.2-gcc-9.2.0, 11.0.2-intel-19.0.4, 11.0.2-gcc-8.3.0, 1.8.0_202-b08-gcc-8.3.0
| openmpi | 3.1.5-gcc-9.2.0, 3.1.5-gcc-8.3.0, 4.0.2-gcc-9.2.0, 3.1.5-gcc-4.8.5
| openssl | 1.1.1d-gcc-8.3.0
| perl | 5.30.0-gcc-9.2.0, 5.30.0-intel-19.0.4, 5.30.0-intel-19.0.5, 5.28.0-gcc-8.3.0, 5.30.0-gcc-7.4.0, 5.30.0-gcc-8.3.0, 5.30.0-gcc-9.2.0, 5.26.2-gcc-4.8.5, 5.30.0-gcc-4.8.5 |
| pgi | 19.4-gcc-4.8.5 |
| picard | 2.19.0-gcc-9.2.0, 2.19.0-intel-19.0.4, 2.19.0-gcc-8.3.0 |
| precice | 1.6.1-gcc-8.3.0-openblas-openmpi
| python | 3.7.4-gcc-9.2.0, 3.7.4-intel-19.0.4, 3.7.4-intel-19.0.5, 2.7.16-gcc-8.3.0, 2.7.16-gcc-9.2.0, 3.7.4-gcc-7.4.0, 3.7.4-gcc-8.3.0, 3.7.4-gcc-4.8.5 |
| quantum-espresso | 6.4.1-intel-19.0.4-impi, 6.4.1-intel-19.0.5-impi |
| r | 1.1.8-gcc-9.2.0, 1.1.8-intel-19.0.4, 3.6.2-gcc-8.3.0-openblas |
| rna-seqc | 1.1.8-gcc-8.3.0 |
| salmon | 0.14.1-gcc-8.3.0 |
| samtools | 1.9-gcc-9.2.0, 1.9-intel-19.0.4, 1.9-gcc-8.3.0, 1.9-gcc-4.8.5 |
| soapdenovo2 | 240-gcc-4.8.5 |
| sratoolkit | 2.9.6-gcc-8.3.0 |
| star | 2.7.0e-gcc-8.3.0 |
| stream | 5.10-intel-19.0.4 |
| stringtie | 1.3.4d-gcc-8.3.0 |
| sundials | 3.1.2-gcc-9.2.0, 4.1.0-gcc-8.3.0-openmpi, 5.0.0-gcc-8.3.0-openmpi |
| tcl | 8.6.8-gcc-8.3.0 |
| tophat | 2.1.2-intel-19.0.4 |
| vardictjava | 1.5.1-gcc-8.3.0 |
| vsearch | 2.4.3-intel-19.0.4, 2.4.3-gcc-8.3.0 |
| vt | 0.5772-gcc-8.3.0 |




## 常用的科学计算软件介绍

| 软件名 | 介绍 |
| ---- | ---- |
| Amber | A package of molecular simulation programs and analysis tools. |
| abinit | ABINIT is a package whose main program allows one to find the total energy, charge density and electronic structure of systems made of electrons and nuclei (molecules and periodic solids) within Density Functional Theory (DFT), using pseudopotentials and a planewave or wavelet basis. ABINIT also includes options to optimize the geometry according to the DFT forces and stresses, or to perform molecular dynamics simulations using these forces, or to generate dynamical matrices, Born effective charges, and dielectric tensors, based on Density-Functional Perturbation Theory, and many more properties. |
| CASTEP | A software package which uses density functional theory to provide a good atomic-level description of all manner of materials and molecules. |
| CESM | Community Earth System Model, or CESM, is a fully-coupled, community, global climate model that provides state-of-the-art computer simulations of the Earth's past, present, and future climate states. |
| CP2K | A freely available program to perform atomistic and molecular simulations of solid state, liquid, molecular and biological systems. It provides a general framework for different methods such as e.g. density functional theory (DFT) using a mixed Gaussian and plane waves approach (GPW), and classical pair and many-body potentials. |
| CPMD | A parallelized plane wave/pseudopotential implementation of Density Functional Theory, particularly designed for ab-initio molecular dynamics. |
| CRYSTAL | CRYSTAL is a general-purpose program for the study of crystalline solids. The CRYSTAL program computes the electronic structure of periodic systems within Hartree Fock, density functional or various hybrid approximations (global, range-separated and double-hybrids). The Bloch functions of the periodic systems are expanded as linear combinations of atom centred Gaussian functions. Powerful screening techniques are used to exploit real space locality. Restricted (Closed Shell) and Unrestricted (Spin-polarized) calculations can be performed with all-electron and valence-only basis sets with effective core pseudo-potentials. |
| GPAW | GPAW is a density-functional theory (DFT) Python code based on the projector-augmented wave (PAW) method and the atomic simulation environment (ASE). |
| GROMACS | GROMACS is a versatile package to perform molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles. It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) many groups are also using it for research on non-biological systems, e.g. polymers. |
| LAMMPS | (Large-scale Atomic/Molecular Massively Parallel Simulator) a classical molecular dynamics code. |
| NAMD | A parallel molecular dynamics application designed to simulate large bio-molecular systems. It is developed and maintained by the University of Illinois at Urbana-Champaign. |
| NWChem | NWChem aims to provide its users with computational chemistry tools that are scalable both in their ability to treat large scientific computational chemistry problems efficiently, and in their use of available parallel computing resources from high-performance parallel supercomputers to conventional workstation clusters. The NWChem software can handle: Biomolecules, nanostructures, and solid-state; From quantum to classical, and all combinations; Gaussian basis functions or plane-waves; Scaling from one to thousands of processors; Properties and relativity. | 
| OpenFOAM | OpenFOAM is an open-source toolbox for computational fluid dynamics. OpenFOAM consists of generic tools to simulate complex physics for a variety of fields of interest, from fluid flows involving chemical reactions, turbulence and heat transfer, to solid dynamics, electromagnetism and the pricing of financial options. The core technology of OpenFOAM is a flexible set of modules written in C++. These are used to build solvers and utilities to perform pre- and post-processing tasks ranging from simple data manipulation to visualisation and mesh processing. |
| Paraview | Paraview is a data visualisation and analysis package. Whilst ARCHER compute or login nodes do not have graphics cards installed in them paraview is installed so the visualisation libraries and applications can be used to post-process simulation data. To this end the pvserver application has been installed, along with the paraview libraries and client application. |
| Quantum Espresso | Quantum Espresso is an integrated suite of Open-Source computer codes for electronic-structure calculations and materials modeling at the nanoscale. It is based on density-functional theory, plane waves, and pseudopotentials. |
| SIESTA | (Scalable Library for Eigenvalue Problem Computations) a library	for parallel computation of eigenvalues and eigenvectors. Based on PETSc. Note: not centrally installed as a module, only build	instructions provided.|
| VASP | A package for ab initio, quantum-mechanical, molecular dynamics simulations. |
| WIEN2k | WIEN2k allows to perform electronic structure calculations of solids using density functional theory. It is based on the full-potential (linearized) augmented plane-wave and local orbitals method, one among the most accurate schemes for band structure calculations. |










