#!/bin/bash -l

# Submit this script as: "./prepare-env.sh" instead of "sbatch prepare-env.sh"

# Prepare user env needed for Slurm batch job
# such as module load, setup runtime environment variables, or copy input files, etc.
# Basically, these are the commands you usually run ahead of the srun command 

module load cray-netcdf
export OMP_NUM_THREADS=4

# Generate the Slurm batch script below with the here document, 
# then when sbatch the script later, the user env set up above will run on the login node
# instead of on a head compute node (if included in the Slurm batch script),
# and inherited into the batch job.

cat << EOF > prepare-env.sl 
#!/bin/bash
#SBATCH -t 30:00
#SBATCH -N 8
#SBATCH -q debug
#SBATCH -C haswell

srun -n 16 -c 32 --cpu_bind=cores ./myapp.exe 

# Other commands needed after srun, such as copy your output filies,
# should still be included in the Slurm script.
cp <my_output_file> <target_location>/.
EOF

# Now submit the batch job
sbatch prepare-env.sl
