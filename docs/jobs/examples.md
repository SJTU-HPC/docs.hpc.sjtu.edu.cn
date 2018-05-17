# Examples

## Basic MPI

One MPI processes per physical core.

??? example "Edison"
	```bash
	--8<-- "docs/jobs/examples/basic-mpi/edison/basic-mpi.sh"
	```

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/basic-mpi/cori-haswell/basic-mpi.sh"
	```

??? example "Cori KNL"
	```bash
	--8<-- "docs/jobs/examples/basic-mpi/cori-knl/basic-mpi.sh"
	```

## Hybrid MPI+OpenMP jobs

One MPI process per socket and 1 OpenMP thread per
physical core

!!! warning
	In Slurm each hyper thread is considered a "cpu" so the
	`--cpus-per-task` option must be adjusted accordingly. Generally
	best performance is obtained with 1 OpenMP thread per physical
	core.

??? example "Edison"
	```bash
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/edison/hybrid-mpi-openmp.sh"
	```

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-haswell/hybrid-mpi-openmp.sh"
	```

??? example "Cori KNL"
	```bash
	--8<-- "docs/jobs/examples/02-hybrid-mpi-openmp/cori-knl/hybrid-mpi-openmp.sh"
	```

## Burst buffer

## Containerized (Docker) applications with Shifter

## MPMD and multi-program jobs

## Core specialization

## Job Arrays

## Dependencies

## OpenMPI

## IntelMPI

## Network topology

## Serial

### Shared QOS

### TaskFarmer

### GNU Parallel

## Interactive
