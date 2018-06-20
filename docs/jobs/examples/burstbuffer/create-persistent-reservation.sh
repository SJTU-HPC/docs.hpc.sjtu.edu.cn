#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#BB create_persistent name=PRname capacity=100GB access_mode=striped type=scratch

