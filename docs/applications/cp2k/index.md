# CP2k

CP2K is a quantum chemistry and solid state physics software package
that can perform atomistic simulations of solid state, liquid,
molecular, periodic, material, crystal, and biological systems.

NERSC provides modules for [cp2k](https://www.cp2k.org).

Use the `module avail` command to see what versions are available:

```bash
nersc$ module avail cp2k
```

## Example - Cori KNL

```bash
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=300
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --core-spec=2

module unload craype-haswell
module load craype-mic-knl
module load cp2k/6.1

srun --cpu-bind=cores cp2k.popt -i example.inp
```

## Support

*  [Reference Manual](https://manual.cp2k.org/)
*  [Forum](https://groups.google.com/group/cp2k)
*  [FAQ](https://www.cp2k.org/faq)
*  [Makefile and build script](https://gitlab.com/NERSC/nersc-user-software/tree/master/cp2k/)

!!! tip
	If *after* consulting with the above you believe there is an
	issue with the NERSC module, please file
	a [support ticket](https://help.nersc.gov).
