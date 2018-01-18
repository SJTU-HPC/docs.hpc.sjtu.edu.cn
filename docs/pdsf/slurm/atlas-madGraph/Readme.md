##How to start ATLAS  **Slurm+Shifter**  job

!!!warning
      This instruction works only for user=kkrizka,  use it as a guidance only.


MadGraph is a  mutli-core compute task, no need for CVMFS


```bash
ssh kkrizka@pdsf
$sbatch oneMG.slr
$  cat oneMG.slr
--8<-- "docs/pdsf/slurm/atlas-madGraph/oneMG.slr"
```
This example will run MadGraph on 6 cores.<br>
 Note, the oneMG.slr is setup to run on all 3 slurm partitions: PDSF+Chos, PDSF+Shifter, Cori+Shifter - you need only to toggle the '-' in front of SBATCH.

The bash task script 'runMad.sh' requires sourcing of your envirement - if you use Shifter
```bash
$  cat runMad.sh
--8<-- "docs/pdsf/slurm/atlas-madGraph/runMad.sh"
```


      
