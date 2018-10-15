# Wannier90

The BerkeleyGW Package is a set of computer codes that calculates the
quasiparticle properties and the optical responses of a large variety
of materials from bulk periodic crystals to nanostructures such as
slabs, wires and molecules. The package takes as input the mean-field
results from various electronic structure codes such as the Kohn-Sham
DFT eigenvalues and eigenvectors computed with Quantum ESPRESSO,
PARATEC, PARSEC, Octopus, Abinit, Siesta etc.

NERSC provides modules for [Wannier90](http://www.wannier.org).

Use the `module avail` command to see what versions are available:

```bash
nersc$ module avail wannier90
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

module load wannier90
srun -n 24 wannier90.x >& test.log
```

## Support

*  [Forum](http://www.wannier.org/forum.html)
*  [Documentation](http://www.wannier.org/user_guide.html)

!!! tip
	If after consulting with the above you believe there is an issue
	with the NERSC module, please file a
	[support ticket](https://help.nersc.gov).