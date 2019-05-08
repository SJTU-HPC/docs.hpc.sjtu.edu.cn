# LAMMPS

[LAMMPS](https://lammps.sandia.gov/) is a large scale classical
molecular dynamics code, and stands for Large-scale Atomic/Molecular Massively
Parallel Simulator.  LAMMPS has potentials for soft materials (biomolecules,
polymers), solid-state materials (metals, semiconductors) and coarse-grained or
mesoscopic systems. It can be used to model atoms or, more generically, as a
parallel particle simulator at the atomic, meso, or continuum scale.

# How to Access LAMMPS

NERSC uses [modules](../../environment/#nersc-modules-environment) to manage
access to software. To use the default version of LAMMPS, type:

```console
module load lammps
```

# Using LAMMPS on Cori

There are two ways of running LAMMPS on Cori: submitting a batch job, or
running interactively in an interactive batch session.

!!! tip "Sample Batch Script to Run LAMMPS on Cori Haswell"
    ```console
    #!/bin/bash
    #SBATCH -J test_lammps
    #SBATCH -C haswell
    #SBATCH -q debug
    #SBATCH -N 2
    #SBATCH -t 30:00
    #SBATCH -o test_lammps.o%j

    module load lammps

    # LAMMPS supports 2 different ways of reading inputs files

    srun -n 64 -c 2 --cpu-bind=cores lmp_cori < test.in
    srun -n 64 -c 2 --cpu-bind=cores lmp_cori -in test.in
    ```

!!! tip "Sample Batch Script to Run LAMMPS on Cori KNL"
    ```console
    #!/bin/bash
    #SBATCH -J test_lammps
    #SBATCH -C knl
    #SBATCH -q debug
    #SBATCH -N 2
    #SBATCH -t 30:00
    #SBATCH -o test_lammps.o%j

    module load lammps

    # LAMMPS supports 2 different ways of reading inputs files

    srun -n 136 -c 2 --cpu-bind=cores lmp_cori < test.in
    srun -n 136 -c 2 --cpu-bind=cores lmp_cori -in test.in
    ```

These job scripts request two nodes in the debug partition, and run for up to
30 minutes. The first example runs 64 MPI processes across 64 cores on 2 nodes
of Cori Haswell. The second example runs 136 MPI processes across 136 cores on
2 nodes of Cori KNL.

Then submit the job script using the sbatch command, e.g., assuming the job
script name is `test_lammps.slurm`:

```console
sbatch test_lammps.slurm
```

# Official LAMMPS documentation



## Support

 * [LAMMPS Online Manual](https://lammps.sandia.gov/doc/Manual.html)

!!! tip
	If *after* the checking the above you believe there is an
	issue with the NERSC module file a ticket with
	our [help desk](https://help.nersc.gov)
