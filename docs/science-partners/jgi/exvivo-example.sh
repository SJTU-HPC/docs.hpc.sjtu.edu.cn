elvis@cori10:~> module load esslurm
elvis@cori10:~> salloc -C skylake -A fungalp -q jgi_interactive
salloc: Granted job allocation 1
salloc: Waiting for resource configuration
salloc: Nodes exvivo006 are ready for job
elvis@exvivo006:~> exit
exit
srun: Terminating job step 1.0
salloc: Relinquishing job allocation 1
elvis@cori10:~> sbatch -C skylake -A fungalp -q jgi_exvivo bioinformatics.sh
Submitted batch job 2
elvis@cori10:~>
