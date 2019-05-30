# Wannier90

[Wannier90](http://www.wannier.org) is a computer package, written in
Fortran90, for obtaining maximally-localised Wannier functions, using
them to calculate bandstructures, Fermi surfaces, dielectric
properties, sparse Hamiltonians and many things besides.

Use the `module avail` command to see what versions are available:

```bash
nersc$ module avail wannier90
```

## Examples

See the [example jobs page](../../jobs/examples/index.md) for additional
examples and information about jobs.

!!! warning
	*Only* the version>=3.0.0 modules have MPI enabled.
	Versions earlier than 3.0.0 should be run in serial with 
	`srun -n 1 ...`.

??? example "Cori Haswell"

	```slurm
	#!/bin/bash
	#SBATCH --constraint=haswell
	#SBATCH --qos=regular
	#SBATCH --time=01:00:00
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=32
	#SBATCH --cpus-per-task=2

	module load wannier90/3.0.0
	srun --cpu-bind=cores wannier90.x
	```

!!! example "Cori KNL"
	```slurm
	#!/bin/bash
	#SBATCH --constraint=knl
	#SBATCH --qos=regular
	#SBATCH --time=01:00:00
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=68
    #SBATCH --cpus-per-task=2
	
 	module load wannier90/3.0.0
	srun --cpu-bind=cores wannier90.x
	```

## Support

A user guide, tutorial, documentation are available at the 
[support](http://www.wannier.org/support) page. Instructions to
report bugs or errors are also provided there.

!!! tip
	If after consulting with the above you believe there is an issue
	with the NERSC module, please file a
	[support ticket](https://help.nersc.gov).
