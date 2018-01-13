##**Interactive  session**  using Shifter on PDSF in SL6.4


```bash
ssh pdsf

salloc -n 1 -p shared  -t 50:00 --image=custom:pdsf-chos-sl64:v4  --volume=/global/project:/project

shifter /bin/bash
export CHOS=sl64
source ~/.bash_profile.ext

cd abc/
```

##How to start DayaBay  **Slurm+Shifter**  job


```bash
ssh pdsf.nersc.gov
$sbatch run1Dyb.slr
$  cat oneMG.slr
--8<-- "docs/pdsf/slurm/dyb-nuwa/run1Dyb.slr"
```
This example will use precopiled NuWa code by Matt - it is avaliable to everyone<br>
 Note, the oneMG.slr is setup to run on all 3 slurm partitions: PDSF+Chos, PDSF+Shifter, Cori+Shifter - you need only to toggle the '-' in front of SBATCH.

The bash task script 'run1DayabayShifter.sh' requires sourcing of your envirement - if you use Shifter
```bash
$  cat run1DayabayShifter.sh
--8<-- "docs/pdsf/slurm/dyb-nuwa/run1DayabayShifter.sh"
```


      
