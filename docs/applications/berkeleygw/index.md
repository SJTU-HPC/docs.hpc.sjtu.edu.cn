# BerkeleyGW

The BerkeleyGW Package is a set of computer codes that calculates the quasiparticle 
properties and the optical responses of a large variety of materials from bulk periodic 
crystals to nanostructures such as slabs, wires and molecules. The package takes as
input the mean-field results from various electronic structure codes such as the 
Kohn-Sham DFT eigenvalues and eigenvectors computed with Quantum ESPRESSO, PARATEC,
PARSEC, Octopus, Abinit, Siesta etc.

NERSC provides modules for [BerkeleyGW](https://www.berkeleygw.org).

Use the `module avail` command to see what versions are available:

```bash
nersc$ module avail berkeleygw
```

## Example

See the [example jobs page](/jobs/examples/) for additional
examples and information about jobs.

### Edison

```
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=2

module load berkeleygw
srun -n 24 epsilon.cplx.x
```

## Support

*  [Forum](https://groups.google.com/a/berkeleygw.org/forum/#!forum/help)
*  [Documentation](https://berkeleygw.org/documentation/)

!!! tip
	If after consulting with the above you believe there is an issue
	with the NERSC module, please file a
	[support ticket](https://help.nersc.gov).