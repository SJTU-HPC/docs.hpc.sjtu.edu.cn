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

## Performance

Performance of cp2k can vary depending on the system size and run type. The 
multinode scaling performance of the code depends on the amount of work (or 
number of atoms) per MPI rank. It is recommended to try a representative 
test case on different number of nodes to see what gives the best performance.

## Support

*  [Reference Manual](https://manual.cp2k.org/)
*  [Forum](https://groups.google.com/group/cp2k)
*  [FAQ](https://www.cp2k.org/faq)
*  [Makefile and build script](https://gitlab.com/NERSC/nersc-user-software/tree/master/cp2k/)

!!! tip
	If *after* consulting with the above you believe there is an
	issue with the NERSC module, please file
	a [support ticket](https://help.nersc.gov).
