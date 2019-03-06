# Quantum ESPRESSO/PWSCF

[Quantum ESPRESSO](https://www.quantum-espresso.org) is an integrated
suite of computer codes for electronic structure calculations and
materials modeling at the nanoscale. It builds on the electronic
structure codes PWscf, PHONON, CP90, FPMD, and Wannier.  It is based
on density-functional theory, plane waves, and pseudopotentials (both
norm-conserving and ultrasoft).

## Using Quantum ESPRESSO at NERSC

NERSC provides modules with precompiled Quantum ESPRESSO
installations.

Available versions can be found by running:

```bash
nersc$ module avail espresso
```

and the specific version is loaded via

```bash
nersc$ module load espresso/<version>
```

### Examples

For all routines except `pw.x`, run QE in full MPI mode as there
is currently no efficient OpenMP implementation available.

!!! example "Cori Haswell"
	```bash
	#!/bin/bash
	#SBATCH -qos=regular
	#SBATCH --nodes=2
	#SBATCH --tasks-per-node=32
	#SBATCH -C haswell
	#SBATCH -t 02:00:00
	#SBATCH -J my_job

	export OMP_NUM_THREADS=1

	module load espresso/6.1
	srun ph.x -input test.in
	```

!!! warning
	Pay close attention to the explicit setting of
	`OMP_NUM_THREADS=1` when running in pure MPI mode. This is optimal
	when intending to run with only MPI tasks.

#### hybrid DFT

We have optimized the hybrid DFT calculations in Quantum ESPRESSO
(`pw.x`). These changes are described in our
[Quantum ESPRESSO case study](../../performance/case-studies/quantum-espresso/index.md)
and available in the `espresso/6.1` module we provide.

The following scripts provides the best `pw.x` performance for hybrid
functional calculations:

!!! example "Cori Haswell"
	```bash
	#!/bin/bash
	#SBATCH --qos=regular
	#SBATCH --nodes=2
	#SBATCH --tasks-per-node=4
	#SBATCH --cpus-per-task=16
	#SBATCH -C haswell
	#SBATCH -t 02:00:00
	#SBATCH -J my_job

	export OMP_NUM_THREADS=8
	export OMP_PLACES=threads
	export OMP_PROC_BIND=spread

	module load espresso/6.1
	srun --cpu_bind=cores pw.x -nbgrp 8 -input test.in
	```

!!! example "Cori KNL"
	```bash
	#!/bin/bash
	#SBATCH --qos=regular
	#SBATCH --nodes=2
	#SBATCH --tasks-per-node=4
	#SBATCH --cpus-per-task=68
	#SBATCH -C knl,quad,cache
	#SBATCH -t 02:00:00
	#SBATCH -J my_job

	export OMP_NUM_THREADS=16
	export OMP_PLACES=threads
	export OMP_PROC_BIND=spread

	module load espresso/6.1
	srun --cpu_bind=cores pw.x -nbgrp 8 -input test.in
	```

!!! tip
	For band-group parallelization, it is recommended to run one
	band group per MPI rank. However, please keep in mind that it is
	not possible to use more band-groups than there are bands in your
	system, so adjust the number accordingly if issues are
	encountered.

!!! note
	The new implementation is much more efficient, so you might
	be able to use much fewer nodes and still get the solution within
	the same wallclock time.

## Compilation Instructions

Some users may be interested in tweaking the Quantum ESPRESSO build
parameters and building QE themselves. Our build instructions for the
QE module are listed below. The following procedure was used to build
Quantum ESPRESSO versions >5.4 on Cori. In the root QE directory:

```bash
nersc$ ./configure
nersc$ cp /usr/common/software/espresso/<version>/<arch>/<comp>/make.inc .
nersc$ make <application-name, e.g. pw>
```

where `<version>` specifies the version, `<arch>` the architecture
(usually `hsw` or `knl` for Haswell and KNL respectively) and `<comp>`
the compiler (usually `gnu` or `intel`).

!!! note
	Not all versions are available for all architectures or
	compilers.
