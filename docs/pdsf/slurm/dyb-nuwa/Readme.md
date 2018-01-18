##How to start DayaBay  **Slurm+Shifter**  job


```bash
ssh pdsf.nersc.gov
$sbatch run1Dyb.slr
$  cat  run1Dyb.slr
--8<-- "docs/pdsf/slurm/dyb-nuwa/run1Dyb.slr"
```
This example will use precopiled NuWa code by Matt - it is avaliable to everyone<br>
 Note, the oneMG.slr is setup to run on all 3 slurm partitions: PDSF+Chos, PDSF+Shifter, Cori+Shifter - you need only to toggle the '-' in front of SBATCH.

The bash task script 'run1DayabayShifter.sh' requires sourcing of your envirement - if you use Shifter
```bash
$  cat run1DayabayShifter.sh
--8<-- "docs/pdsf/slurm/dyb-nuwa/run1DayabayShifter.sh"
```


      
