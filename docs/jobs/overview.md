# Running jobs

> :bulb: **Key points**
>
>  * When you login to a NERSC cluster you land on a *login node*. Login nodes 
>    are for editing, compiling, preparing jobs, etc. Don't run jobs on the 
>    login nodes, instead, submit them to the batch system
>
>  * The batch system (Slurm) gives your job access to compute nodes. You can 
>    submit a job script to the batch system with `sbatch my-job-script.sh`
>
>  * The job script runs once, on one of the compute nodes allocated to your 
>    job. To start multiple instances of a command across multiple nodes, use
>    `srun the-command-to-run`

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore
et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut a
liquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillu
m dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui
 officia deserunt mollit anim id est laborum.
