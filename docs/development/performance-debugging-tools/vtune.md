# VTune

## Using VTune

See the [Intel VTune Amplifier documentation](https://software.intel.com/en-us/vtune-amplifier-help) for general usage.

VTune is available on all NERSC production systems by loading the VTune module.

```shell
nersc$ module load vtune
```

It is generally recommended to defer finalization when running on KNL. Finalization is an inherently serial process and
the individual core performance on KNL is very poor. Thus, when running VTune on KNL, add the parameter `-finalization-mode=deferred`:

```shell
#!/bin/bash

#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -t 00:30:00
# ... additional sbatch parameters ...

module load vtune

amplxe-cl -finalization-mode=deferred -collect ... -r <result-dir> -- <command-to-profile>

# in some cases, it one might want to copy over the libraries need to finalize
amplxe-cl -archive -r <result-dir>
```

and then finalize on a login (Haswell) node:

```Shell
nersc$ amplxe-cl -finalize -result-dir <PATH>
```

## Using VTune with Shifter

VTune can be attached to a Shifter container by executing the process in the background and then attaching VTune to the process via the PID (process identifier).
The following will not work:

```shell
amplxe-cl -collect ... -- shifter <command-to-execute-in-container>
```

The recommended method is as follows:

```Shell
#!/bin/bash

#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --image=<username/some-image>
# ... additional sbatch parameters ...

module load vtune

PID_FILE=$(mktemp pid.XXXXXXX)

# the first "&" causes the command to execute in the background
# "echo $!" prints the PID
# "&> ${PID_FILE}" writes the PID to the temporary file
shifter <command-to-execute-in-container> & echo $! &> ${PID_FILE}

# read the PID from the file
TARGET_PID=$(cat ${PID_FILE})

# attach VTune to the process
amplxe-cl -collect <collection-mode> --target-pid=${TARGET_PID} ...

```

### VTune finalization with Shifter

In the [Using VTune](#using-vtune) section, it was recommended to not finalize on KNL. However, when using containers, deferring finalization creates a problem
because the binaries needed for finalization exist only within the container. Due to this fact, it is recommended to not defer finalization.

## VTune + Shifter Example

```Shell
#!/bin/bash

#SBATCH --qos=regular
#SBATCH --constraint=knl
#SBATCH -A dasrepo
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH --job-name=tomopy_gridrec
#SBATCH --output=out_tomopy_%j.log
#SBATCH --image=jrmadsen/tomopy-reference:gcc

set -o errexit

# ensure VTune module is loaded
module load vtune

# this format of assignment only sets the variable to the specified value
# if not already set in the environment
: ${OMP_NUM_THREADS:=1}
: ${NUMEXPR_MAX_THREADS:=$(nproc)}
: ${VTUNE_COLLECTION_MODE:="advanced-hotspots"}
: ${VTUNE_SAMPLING_INTERVAL:=25}
: ${VTUNE_RESULTS_DIR:=$(mktemp -d ${PWD}/run-${VTUNE_COLLECTION_MODE}-XXXXX)}

export OMP_NUM_THREADS
export NUMEXPR_MAX_THREADS
export VTUNE_COLLECTION_MODE
export VTUNE_SAMPLING_INTERVAL
export VTUNE_RESULTS_DIR

# make sure empty, let vtune create directory
rm -rf ${VTUNE_RESULTS_DIR}

# use mktemp to ensure guard against multiple jobs in same dir
PID_FILE=$(mktemp pid.XXXXXX)

echo -e "\n### Submitting shifter job into background and storing PID in file: ${PID_FILE} ###\n"
shifter /opt/conda/bin/python ./run_tomopy.py -a gridrec -n 256 -s 512 -f jpeg -S 1 -c 8 -p shepp3d -i 5 & echo $! &> ${PID_FILE}

echo -e "\n### Reading PID file: ${PID_FILE} ###\n"
TARGET_PID=$(cat ${PID_FILE})

# echo the ps for debugging
echo -e "\n### Target PID: ${TARGET_PID} ###\n"
ps

# echo the environment for reference
echo -e "\n### Environment ###\n"
env

echo -e "\n### Attaching VTune process to PID ${TARGET_PID} ###\n"
amplxe-cl \
    -collect ${VTUNE_COLLECTION_MODE} \
    -knob collection-detail=hotspots-sampling \
    -knob event-mode=all \
    -knob analyze-openmp=true \
    -knob sampling-interval=${VTUNE_SAMPLING_INTERVAL} \
    -data-limit=0 \
    --target-pid=${TARGET_PID} \
    -r ${VTUNE_RESULTS_DIR}

echo -e "\nCompleted\n"
```