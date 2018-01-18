##How to start Majorana  **Slurm+Shifter**  job executing arbitrary MJD code


```bash
ssh pdsf.nersc.gov
$sbatch batchShifter.slr
$  cat batchShifter.slr
--8<-- "docs/pdsf/slurm/majorana-skim/batchShifter.slr"
```
This example need adjustment of paths for every user.

The bash task script 'inShifterJob.sh' requires sourcing of your envirement - if you use Shifter
```bash
$  cat .inShifterJobsh
--8<-- "docs/pdsf/slurm/majorana-skim/inShifterJob.sh"
```

You can start Slurm job like this:
```bash
sbatch batchShifter.slr './skim_mjd_data 1 1'
```



      
