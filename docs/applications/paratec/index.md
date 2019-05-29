# PARATEC

PARATEC is a parallel, plane-wave basis, density functional theory
(DFT) code developed at Berkeley.  PARATEC is one of the DFT packages
supported by the BerkeleyGW code. PARATEC supports many traditional
DFT features and exchange-correlation functionals. PARATEC uses
norm-conserving pseudopotentials that can be generated with the FHI
pseudopotential program.

NERSC provides modules for PARATEC.

Use the `module avail` command to see what versions are available:

```bash
nersc$ module avail paratec
```

## Example

See the [example jobs page](../../jobs/examples/index.md) for additional
examples and information about jobs.

### Cori

```slurm
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --constraint=haswell

module load paratec
srun paratec.mpi
```

## Support

*  [BerkeleyGW Forum](https://groups.google.com/a/berkeleygw.org/forum/#!forum/help)
*  [Documentation](http://oldsite.berkeleygw.org/releases/manual_v1.2.0.html#MeanField/PARATEC/README)

!!! tip
	If after consulting with the above you believe there is an issue
	with the NERSC module, please file a
	[support ticket](https://help.nersc.gov).
