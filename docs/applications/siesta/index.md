# SIESTA

SIESTA is both a method and its computer program implementation, to
perform efficient electronic structure calculations and ab initio
molecular dynamics simulations of molecules and solids. SIESTA's
efficiency stems from the use of strictly localized basis sets and
from the implementation of linear-scaling algorithms which can be
applied to suitable systems. A very important feature of the code is
that its accuracy and cost can be tuned in a wide range, from quick
exploratory calculations to highly accurate simulations matching the
quality of other approaches, such as plane-wave and all-electron
methods.

## Example run scripts

!!! example "Cori KNL"
	```slurm
	#!/bin/bash
	#SBATCH --qos=regular
	#SBATCH --time=01:30:00
	#SBATCH --nodes=2
	#SBATCH --tasks-per-node=68
	#SBATCH --constraint=knl

	module load siesta
	srun siesta < test.fdf > test.out
	```

??? example "Cori Haswell"
    ```slurm
	#!/bin/bash
	#SBATCH --qos=regular
	#SBATCH --time=01:30:00
	#SBATCH --nodes=2
	#SBATCH --tasks-per-node=32
	#SBATCH --constraint=haswell

	module load siesta
	srun siesta < test.fdf > test.out
    ```

## Support

* [Manuals](https://departments.icmab.es/leem/siesta/Documentation/Manuals/manuals.html)
* [Development and distribution platform](https://launchpad.net/siesta)
    * [Questions](https://answers.launchpad.net/siesta)
    * [FAQ](https://answers.launchpad.net/siesta/+faqs)
    * [Bug reports](https://bugs.launchpad.net/siesta)
* If *after* consulting the above you believe there is an issue with
  the NERSC module/installation file a ticket with
  our [help desk](https://help.nersc.gov)
