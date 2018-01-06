#!/bin/bash -l
#   A Slurm batch job is just a script, decorated with #SBATCH directives
#   telling Slurm what it needs to know to schedule this job. We recommend 
#   using bash, but other scripting languages such as csh or python also work.
#   As with any script, the first line should begin with 
#   '#!/path/to/interpreter', which tell Unix what to do with this script.
#
#   The "-l" flag tells bash to read your $HOME/.bash_profile so that
#   anything you have added to $HOME/.bash_profile.ext will appear in
#   your script environment

#SBATCH --nodes=2            # or -N 2
#SBATCH --time=00:10:00      # or -t 10
#SBATCH --constraint=knl     # or -C knl
#SBATCH --license=SCRATCH    # or -L SCRATCH
#SBATCH --qos=regular        # or -q regular
##SBATCH --mem=20GB          # only needed with -q shared
#   The first section of a Slurm job script has directives for Slurm, 
#   indicated by '#SBATCH' at the start of the line (note that '#' indicates
#   a comment in scripting languages such as bash, so the Slurm directive is
#   a specially-formatted comment). Don't add any whitespace in or before 
#   '#SBATCH', or Slurm won't recognise it. Slurm also ignores any directives
#   after the first non-comment, non-blank line in the script 
#
#   Note that the directives can have trailing comments, and that the 
#   directive itself can be commented out
#
#   Most #SBATCH directives have a short form, which we have included above
#   in the trailing comment. #SBATCH directives can also be specified on the
#   command line, eg:
#       sbatch -t 10 my_job_script.sh
#   If you specify the same option as aa directive and on the command line, 
#   the one on the command line takes precedence
#
#   At NERSC we require each job to specify:
#    - the number of nodes needed (--nodes)
#    - the time (real wallclock time) after which Slurm can kill the job
#      if it has not yet finished (--time)
#    - the type of nodes needed (--constraint)
#    - the filesystems needed (--licence)
#    - the policy set (approximately "queue") to submit to
#  Each of these is explained in more detail in our web pages

export RUNDIR=$SCRATCH/run-$SLURM_JOBID
mkdir -p $RUNDIR
cd $RUNDIR
#   You will mostly store job scripts in $HOME but the job itself should run
#   in a $SCRATCH or $DW (burst buffer) filesystem - $HOME is not equipped to
#   handle the I/O load of running jobs. Slurm sets a number of environment 
#   variables with information about the current job, so the above idiom 
#   creates a unique directory for this job to run in

srun -n 4 bash -c 'echo "Hello, world, from node $(hostname)"' # >stdout 2>&1
#   The job script runs only once on the first node allocated to your job. To 
#   run a command in multiple tasks spread over all of your job's nodes, use
#   srun. In this example our command is a simple inline bash script that we
#   will run 4 copies of, 2 on each node
#   Any stdout and stderr from the tasks will be sent back to the first node 
#   and written to stdout and stderr, which is saved in a file with a name
#   like 'slurm-12345.out', in the directory you submitted the job from. If 
#   your job might produce reams of stdout and stderr this could overwhelm 
#   $HOME so you should redirect it to a file in your run directory, as shown
#   in the trailing comment here


