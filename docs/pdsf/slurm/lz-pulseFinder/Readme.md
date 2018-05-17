## **Interactive  session**  using Shifter on PDSF in SL6.4

!!!warning
	This instruction works only for user=afan, use it as a
	guidance only.

```bash
ssh afan@pdsf

salloc -n 1 -p shared  -t 50:00 --image=custom:pdsf-chos-sl64:v4  --volume=/global/project:/project

shifter /bin/bash
export CHOS=sl64
source ~/.bash_profile.ext

cd forJan/PhotonDetection/
source /global/project/projectdirs/lz/releases/physics/latest/Physics/setup.sh
lzap_project
time lzap scripts/validations/PulseFinderValidation.py
```

!!! note
	You do not need to load ROOT, CLHEP, or Geant4. This
	application gets all its dependencies from cvmfs, including ROOT.

!!! note
	The TCling error from lzap is expected for now.

<hr>

## How to start LZ **Slurm+Shifter+CVMFS** job

```bash
ssh -A -Y afan@pdsf
$ sbatch lzOne.slr

$ cat hello2.slr
--8<-- "docs/pdsf/slurm/lz-pulseFinder/lzOne.slr"
```

where 'lzOne.slr' informs SLURM what resources will you need and
launches the bash script running the actuall task 'lzReco.sh'. Note,
the lzOne.slr is setup to run on all 3 slurm partitions: PDSF+Chos,
PDSF+Shifter, Cori+Shifter - you need only to toggle the '-' in front
of SBATCH.

The bash task script 'lzReco.sh' requires sourcing of your
envirement - if you use Shifter

```bash
$cat lzReco.sh
--8<-- "docs/pdsf/slurm/lz-pulseFinder/lzReco.sh"
```

The output file for this job was:

```bash
$ sbatch  lzOne.slr
   Submitted batch job 2111

$cat slurm-2111.out
<snip>
Adding to following projects to this work space : PhotonDetection
# setting LC_ALL to "C"
# --> Including file '/global/u2/a/afan/forJan/PhotonDetection/scripts/validations/PulseFinderValidation.py'
# <-- End of file '/global/u2/a/afan/forJan/PhotonDetection/scripts/validations/PulseFinderValidation.py'
ApplicationMgr    SUCCESS
<snip>
LdrfContext          INFO Opening file: /project/projectdirs/lz/data/simulations/LUXSim_release-4.4.6_geant4.9.5.p02/full_slow_simulation/electron_recoils/FullSlowSimulation_ER_flat_10k_DER.root
<snip>
EventLoopMgr         INFO Histograms converted successfully according to request.
ApplicationMgr       INFO Application Manager Finalized successfully
ApplicationMgr       INFO Application Manager Terminated successfully

real	8m31.548s
user	5m25.573s
sys	0m5.361s
returned from cgroup ----
end-A
```
