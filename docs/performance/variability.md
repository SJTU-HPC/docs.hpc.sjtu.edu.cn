# Performance variability

There are many potential sources of variability on an HPC system and
NERSC has identified the following best practices to mitigate
variability and improve application performance.

## hugepages

Use of hugepages can reduce the cost of accessing memory, especially
in the case of many `MPI_Alltoall` operations.

1. load the hugepages module (`module load craype-hugepages2M`)
1. recompile your code
1. add `module load craype-hugepages2M` to batch scripts

!!! note
	Consider adding `module load craype-hugepages2M` to
	`~/.bashrc.ext`.

For more details see the manual pages (`man intro_hugepages`) on Cori
or Edison.

## Location of executables

Compilation of executables should be done in `$HOME` or
`/tmp`. Executables can be copied into the compute node memory at the
start of a job with `sbcast` to greatly improve job startup times and
reduce run-time variability in some cases:

```bash
sbcast -f --compress ./my_program.x /tmp/my_program.x
srun -n 1024 -c 2 --cpu-bind=cores /tmp/my_program.x
```

For applications with dynamic executables and many libraries
(*especially* python based applications)
use [shifter](/programming/shifter/overview.md).

## Topology

Cori and Edison both use a Cray Aries network with a Dragonfly
topology. Slurm has some options to control the placement of parallel
jobs in the system. A "compact" placement can isolate a job from other
traffic on the network.

Slurm's topology awareness can enabled by adding --switches=N to your
job script, where N is the number of switches for your job. Each
"switch" corresponds to approximately 384 compute nodes. This is most
effective for jobs using less than ~300 nodes and 2 or fewer
switches. Note: requesting fewer switches may increase the time it
takes for your job to start.

```
#SBATCH -N 256
#SBATCH --switches=2
```

## Affinity

Running with correct affinity and binding options can greatly affect
variability.

*  use at least 8 ranks per node (1 rank per node cannot utilize the
   full network bandwidth)
*  read `man intro_mpi` for additional options
*  check
   [job script generator](https://my.nersc.gov/script_generator.php) to
   get correct binding
*  use check-mpi.*.cori and check-hybrid.*.cori to
   check affinity settings

```bash
user@nid01041:~> srun -n 8 -c 4 --cpu-bind=cores check-mpi.intel.cori|sort -nk 4
Hello from rank 0, on nid07208. (core affinity = 0,68,136,204)
Hello from rank 1, on nid07208. (core affinity = 1,69,137,205)
Hello from rank 2, on nid07208. (core affinity = 2,70,138,206)
Hello from rank 3, on nid07208. (core affinity = 3,71,139,207)
Hello from rank 4, on nid07208. (core affinity = 4,72,140,208)
Hello from rank 5, on nid07208. (core affinity = 5,73,141,209)
Hello from rank 6, on nid07208. (core affinity = 6,74,142,210)
Hello from rank 7, on nid07208. (core affinity = 7,75,143,211)
```

## Core specialization

Using core-specialization (`#SBATCH -S n`) moves OS functions to
cores not in use by user applications, where n is the number of cores
to dedicate to the OS. The example shows 4 cores per node on KNL for
the OS and the other 64 for the application.

```bash
#SBATCH -S 4
srun -n 128 -c 4 --cpu-bind=cores /tmp/my_program.x
```

## Combined example

This example is for Cori KNL.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --constraint=knl,quad,cache
#SBATCH --qos=regular
#SBATCH --time=60
#SBATCH --core-spec=4
#SBATCH --switches=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4

module load craype-hugepages2M

sbcast -f  --compress ./my_program.x /tmp/my_program.x
srun --cpu-bind=cores /tmp/my_program.x
```

