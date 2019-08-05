#!/bin/bash -l

# Submit this script as: "./prepare-env.sh" instead of "sbatch prepare-env.sh"

# Prepare user env needed for Slurm batch job
# such as module load, compile your code, setup runtime environment variables, etc.
# Basically, these are the commands you usually run ahead of the srun command 

# module load cray-netcdf
cc -qopenmp -o xthi xthi.c
export OMP_NUM_THREADS=4

# Generate the Slurm batch script below with the here document, 
# then when sbatch the script later, the user env set up above will run on the login node
# instead of on a head compute node (if included in the Slurm batch script),
# and inherited into the batch job.
# Notice other_commands_needed_after_srun should still be incldued in the Slurm script.

cat << EOF > prepare-env.sl 
#!/bin/bash
#SBATCH -t 30:00
#SBATCH -N 8
#SBATCH -q debug
#SBATCH -C haswell

srun -n 16 -c 32 --cpu_bind=cores ./xthi 

# add other commands after the srun here, such as archive run output
cat prepare-env.sl
ls -al xthi
EOF

# Now submit the batch job
sbatch prepare-env.sl
