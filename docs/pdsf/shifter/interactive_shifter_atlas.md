# Interactive Shifter

This instruction shows how to use Shifter interactively for PDSF and
Cori, uses the CHOS-SL6.4 image, CVMFS, features Atlas simulation job,
runs for any PDSF user.

1.  Launch Shifter image at the system of your choice, this step is
    different for each system.

    **PDSF  'normal' shifter**

    ```bash
    salloc -p debug  -t 25:00
    echo inShifter:`env|grep  SHIFTER_RUNTIME`
    shifter --image=custom:pdsf-chos-sl64:v4   --volume=/global/project:/project
    echo inShifter:`env|grep  SHIFTER_RUNTIME`
    ```

    The correct output from the 2nd echo is:
    inShifter:**`SHIFTER_RUNTIME=1`**

    **PDSF  cori-like OS**

    ```bash
    salloc -p nucori -n2  -t 25:00
    shifter --image=custom:pdsf-chos-sl64:v4  --module=cvmfs  --volume=/global/project:/project
    echo inShifter:`env|grep  SHIFTER_RUNTIME`
    ```

    The `-n2` limits number of vCores to 2.

    **Cori login node**  (no salloc)

    ```bash
    shifter --image=custom:pdsf-chos-sl64:v5  --module=cvmfs  --volume=/global/project:/project --volume=/global/project/projectdirs/mpccc/balewski/tmp1-atlas:/tmp bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/projectdirs/mpccc/balewski/atlas_input/
    ```

    The LD_LIB-line is needed because Atlas software wants
    libssl.so.10 libcrypto.so.10 which do not exist in the Shifter
    image. The /tmp maping is needed because Atlas jobs want to writ
    ethere and w/o the mapping /tmp would be carved out out of
    RAM.

2.  Initialize CVMFS-based setup for Atlas simulation (should work for
    any PDSF user)

    ```bash
    cd junkDir1  # a lot of output files will be created

    # define shortcuts
    numEve=1
    inpFile=/project/projectdirs/mpccc/balewski/atlas_input/EVNT.01416911._000004.pool.root.1
    verRelease=21.0.65,Athena
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase

    #run setup scripts, takes few seconds
    time source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
    time asetup $verRelease
    ```

3.  Run Atlas simulation job, takes ~15 min 1st time, ~8 minutes 2nd
    time

    ```bash
    # Patch environment for using the DB replica on CVMFS
    export G4ATLAS_SKIPFILEPEEK=1
    export CORAL_AUTH_PATH=/cvmfs/atlas.cern.ch/repo/sw/database/DBRelease/current/XMLConfig
    export CORAL_DBLOOKUP_PATH=/cvmfs/atlas.cern.ch/repo/sw/database/DBRelease/current/XMLConfig
    export DATAPATH=/cvmfs/atlas.cern.ch/repo/sw/database/DBRelease/current:$DATAPATH
    unset FRONTIER_SERVER

    # Run the simulations
    time Sim_tf.py --inputEVNTFile $inpFile --checkEventCount=False --outputHitsFile myHITS.pool.root --maxEvents=$numEve --randomSeed=26741007 --DataRunNumber=222222 --conditionsTag=OFLCOND-RUN12-SDR-01 --enableLooperKiller=True --simulator=MC12G4 --useISF=True --geometryVersion=ATLAS-R2-2016-01-00-01_VALIDATION --ignorePatterns='ToolSvc.ISFG4.+ERROR\s+ISF_to_G4Event.+article.conversion.failed' --preInclude 'SimulationJobOptions/preInclude.BeamPipeKill.py'
    ```

    The correct output is: ...  stopped at Mon Dec 10 11:18:16 2018,
    **trf exit code 0**
