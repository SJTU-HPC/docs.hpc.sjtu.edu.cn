# VASP

VASP is a package for performing ab initio quantum-mechanical
molecular dynamics (MD) using pseudopotentials and a plane wave basis
set. The approach implemented in VASP is based on a finite-temperature
local-density approximation (with the free energy as variational
quantity) and an exact evaluation of the instantaneous electronic
ground state at each MD step using efficient matrix diagonalization
schemes and an efficient Pulay mixing.

## Access

VASP is available only to NERSC users who already have an existing
VASP license.  If you have a VASP license, send your license
information to the VASP support at Vienna at
vasp.materialphysik@univie.ac.at (and copy <vasp_licensing@nersc.gov>)
requesting that they confirm your VASP license to NERSC staff at
vasp_licensing@nersc.gov in order to gain access to the VASP binaries
at NERSC.

!!! note
	NERSC does not provide the VASP 4 modules anymore. 

It may take several business days from the date when your confirmation
request is sent to the VASP support at Vienna to you actually gain the
access to the VASP binaries provided at NERSC. If the confirmation
takes longer than 5 business days, update the initial email thread with
the license information.

When your VASP license is confirmed, NERSC will add you to a unix file
group: `vasp5` for VASP 5. You can check if you have
the VASP access at NERSC or not by typing the `groups` command. If you
are in the `vasp5` file group, then you can access VASP 5 binaries provided at NERSC. 

## Modules

We provide multiple VASP builds for users. To see what VASP modules are available:
```shell
cori$ module avail vasp
```
For example, these are the available modules (as of 7/8/2019),

```shell
cori$ module avail vasp

------------------------- /usr/common/software/modulefiles --------------------------
vasp/5.4.1-hsw                    vasp/20170629-hsw      vasp-tpc/5.4.1-hsw
vasp/5.4.1-knl                    vasp/20170629-knl      vasp-tpc/5.4.1-knl
vasp/5.4.4-hsw(default)           vasp/20171017-hsw      vasp-tpc/5.4.4-hsw(default)
vasp/5.4.4-knl                    vasp/20171017-knl      vasp-tpc/5.4.4-knl
vasp/20170323_NMAX_DEG=128-hsw    vasp/20181030-hsw      vasp-tpc/20170629-hsw
vasp/20170323_NMAX_DEG=128-knl    vasp/20181030-knl      vasp-tpc/20170629-knl
```

Where the modules with "5.4.4" or "5.4.1" in their version strings are the builds of the pure MPI VASP codes, 
and the modules with "2018" or "2017" in their version strings are the builds of the hybrid MPI+OpenMP VASP codes. 
The vasp-tpc (tpc stands for third party codes) modules are the custom builds incorporating commonly used 
third party contributed codes, e.g., the VTST code from University of Texas at Austin,  Wannier90, BEFF, VASPSol, etc.. 
The "knl" and "hsw" in the version strings indicate the modules are optimal builds for Cori KNL and Haswell, respectively. 
The current default on Cori is vasp/5.4.4-hsw (release date: April 17, 2017, with the latest [patches](https://www.vasp.at/index.php/news)).
and you can access it by 

```shell
cori$ module load vasp
```
To use other non-default module, you need to provide the full module name, 
```shell
cori$ module load vasp/20181030-knl
```
The "module show" command shows what VASP modules do to your environment, e.g. 

```shell
cori$ module show vasp/20181030-knl
-------------------------------------------------------------------
/usr/common/software/modulefiles/vasp/20181030-knl:

module		 load craype-hugepages2M 
module-whatis	 VASP: Vienna Ab-initio Simulation Package
This is the vasp-knl development version (last commit 10/30/2018). Wannier90 v1.2 was enabled in the build.
 
setenv		 PSEUDOPOTENTIAL_DIR /usr/common/software/vasp/pseudopotentials/5.3.5 
setenv		 VDW_KERNAL_DIR /usr/common/software/vasp/vdw_kernal 
setenv		 NO_STOP_MESSAGE 1 
setenv		 MPICH_NO_BUFFER_ALIAS_CHECK 1 
setenv		 MKL_FAST_MEMORY_LIMIT 0 
setenv		 OMP_STACKSIZE 256m 
setenv		 OMP_PROC_BIND spread 
setenv		 OMP_PLACES threads 
prepend-path	 PATH /usr/common/software/vasp/vtstscripts/3.1
prepend-path	 PATH /global/common/cori/software/vasp/20181030/knl/intel/bin 
-------------------------------------------------------------------
```

This vasp module adds the path to the VASP binaries to your search
path, and also sets a few environment variables.  Where
`PSEUDOPOTENTIAL_DIR` and `VDW_KERNAL_DIR` are defined for the
locations of the pseudopotential files and the `vdw_kernel.bindat`
file used in dispersion calculations. The OpenMP and MKL environment
variables are set for optimal performance.

## Vasp binaries

Each VASP module provides the three different binaries:

* `vasp_gam` - gamma point only build
* `vasp_ncl` - non-collinear spin
* `vasp_std` - the standard kpoint binary

You need to choose an appropriate binary to run your job.

## Running batch jobs 

To run batch jobs, you need to prepare a job script (see samples
below), and submit it to the batch system with the "sbatch"
command. Assume the job script is named as run.slurm,

```shell
cori$ sbatch run.slurm
```

Please check the [Queue Policy](../../jobs/policy.md) page for the
available QOS's and their resource limits.
 
### Cori Haswell

1. A sample job script to run the **Pure MPI VASP** codes

   ```slurm
   --8<-- "docs/applications/vasp/examples/cori-haswell-simple.sh"
   ```

2. A sample job script to run the **hybrid MPI+OpenMP VASP** codes

   ```slurm
   --8<-- "docs/applications/vasp/examples/cori-haswell-hybrid.sh"
   ```

### Cori KNL

1. A sample job scripts to run the **Pure MPI VASP** codes

   ```slurm
   --8<-- "docs/applications/vasp/examples/cori-knl-simple.sh"
   ```

2. A sample job scripts to run the **hybrid MPI+OpenMP VASP** codes

   ```slurm
   --8<-- "docs/applications/vasp/examples/cori-knl-hybrid.sh"
   ```

!!! Tips
    1. For a better job throughput, run jobs on Cori KNL. 
    2. The hybrid MPI+OpenMP VASP are recommended on Cori KNL for optimal performance.
    3. More performance tips can be found in a [Cray User Group 2017 proceeding](https://cug.org/proceedings/cug2017_proceedings/includes/files/pap134s2-file1.pdf)
    4. Also refer to the presentation [slides](https://www.nersc.gov/users/training/events/vasp-user-hands-on-knl-training-june-18-2019/) for the VASP user training (6/18/2019).

## Runninge interactively

To run VASP interactively, you need to request a batch session using
the "salloc" command, e.g., the following command requests one **Cori
Haswell** node for one hour,

```shell
cori$ salloc -N 1 -q interactive -C haswell -t 1:00:00
```

When the batch session returns with a shell prompt, execute the following commands: 

```shell
cori$ module load vasp 
cori$ srun -n32 -c2 --cpu-bind=cores vasp_std
```
To run on **Cori KNL** interactively, do

```shell
cori$ salloc -N 2 -q interactive -C knl -t 4:00:00
```

The above command requests two KNL nodes for four hours. When the
batch session returns with a shell prompt, execute the following
commands:

```shell
cori$ module load vasp/20181030-knl
cori$ export OMP_NUM_TRHEADS=4
cori$ srun -n32 -c16 --cpu-bind=cores vasp_std
```

!!! Tips
    1. The interactive QOS allocates the requested nodes immediately
       or cancels your job in about 5 minutes (when no nodes are available).
       See the [Queue Policy](../../jobs/policy.md) page for more info.
    2. Test your job using the interactive QOS before submitting a
       long running job.

## VASP makefiles

If you need to build VASP by yourselves, use the makefiles available
at the VASP installation directories.  For example, the
"makefile.include" file that we used to build the vasp/5.4.4-hsw
module is available at,

```
/global/common/sw/cray/cnl6/haswell/vasp/5.4.4/intel/17.0.2.174/4bqi2il 
```

Type `module show `<a vasp module>`` to find the installation directory.

## Documentation

[VASP Online Manual](http://cms.mpi.univie.ac.at/vasp/vasp/vasp.html)
