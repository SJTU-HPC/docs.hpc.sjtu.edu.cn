#!/bin/bash

set -e
set -x

make clean
make

if [ $NERSC_HOST == edison ]; then
    sbatch -W run-edison.sh
fi
