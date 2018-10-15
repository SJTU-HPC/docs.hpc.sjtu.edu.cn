## How to start STAR  **Slurm+Shifter**  job

The root4star reads raw data file and reconstructs events. It requires
acccess to STAR DB. This example will connect to LBL mirrir of STAR
DB.

The Slurm job definition 'starOne.slr' will run on the 4 NERSC
susbsystems:

* PDSF w/ CHOS
* PDSF w/ Shifter
* Cori w/ Shifter
* Edison w/ Shifter

To select the righ version you need to remove '-' sign from *#-SBTACH
bhla* line(s). The other 3 lines will be ignord by Slurm

```bash
ssh  ssh  -X pdsf.nersc.gov
$ sbatch starOne.slr
$  cat starOne.slr
--8<-- "docs/pdsf/slurm/star-bfc/starOne.slr"
```

This example will run root4star and store finle muDst in a different
location than all auxiliary root4star output.

The tcsh task script 'r4sTask_bfc.csh' looks like this

```bash
$  cat r4sTask_bfc.csh
--8<-- "docs/pdsf/slurm/star-bfc/r4sTask_bfc.csh"
```

## How to start STAR  job array using **Slurm+Shifter**

The same 'starOne.slr' can be used to launch an array of jobs, each
processing a different file from a 'dataListB.txt'

```bash
$  cat dataListB.txt
--8<-- "docs/pdsf/slurm/star-bfc/dataListB.txt"
```

To make this work you need to enable fire the same job as array,
counting from 0 to N-1:

```bash
ssh  ssh  -X pdsf.nersc.gov
$ sbatch --array 0-9 starOne.slr
```

which will result with submission 10 independent Slurm jobs, each will
have assigned a different line to the variable 'dataName' passed as an
argument to 'starOne.slr'
