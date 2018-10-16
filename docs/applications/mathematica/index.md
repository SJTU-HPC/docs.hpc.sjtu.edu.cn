# Mathematica

## ![Mathematica logo](./images/mathematica-spikey.png)<br/>Description and Overview

Mathematica is a fully integrated environment for technical computing. It performs symbolic manipulation of equations, integrals, differential equations, and most other mathematical expressions. Numeric results can be evaluated as well.

## How to Use Mathematica

### Running in the Notebook Interface

To use the graphical interface to Mathematica, you will need to connect to a NERSC machine with X11.

```shell
mylaptop$ ssh -X edison.nersc.gov
```
We highly recommend the use of [NX](https://www.nersc.gov/users/connecting-to-nersc/using-nx/) to invoke an X11 session and run the Mathematica notebook interface.

Next, you will need to load the Mathematica module. To use the default version of Mathematica, use

```shell
nersc$ module load mathematica
```

To start up Mathematica once the module is loaded, you can simply type

```shell
nersc$ mathematica
```

You should see the Mathematica logo ("Spikey") appear, followed by the application interface.

### Mathematica Licensing

NERSC's Mathematica licenses are for a limited number of seats. Once you are finished using Mathematica, please be sure you disconnect your X11 session so that your seat becomes available to the next user. If you use NX, remember to exit Mathematica within your NX session before closing, as NX will otherwise keep the session alive, preventing other users from launching Mathematica.

When starting up a new Mathematica session, check to be sure that you don't already have an instance of Mathematica running. The most common issue with Mathematica licensing at NERSC is that another user is inadvertently using multiple seats.

### Running Mathematica Scripts

To run Mathematica scripts, you can do so in interactive mode or in batch mode. Both approaches require the use of the job scheduler. On both Cori and Edison, this is [SLURM](https://www.nersc.gov/users/computational-systems/cori/running-jobs/slurm-at-nersc-overview/).

To run in interactive mode, use salloc to obtain access to a compute node. To avoid using multiple license seats at once (always a good thing), specify a single node and a single task per node. If you want to take advantage of parallelism in your script, you can specify more than one cpu per task. An allocation to run on four cores in the regular queue would be obtained with a command like the following:

```shell
nersc$ salloc  -N 1 -n 1 -c 4 -p regular -t 00:10:00
```
Running the script is then as simple as

```shell
nersc$ srun ./mymathscript.m
```

To run in batch mode, you will need to write a SLURM batch script and then use sbatch to put your job into the queue. An example batch script that makes 4 cores available to your script follows.

```shell
#! /bin/bash -l

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:02:00
#SBATCH -n 1
#SBATCH -c 4

module load mathematica
srun ./mersenne-cori.m
```

If you want to take advantage of parallelism in Mathematica, you can use the application's built-in parallel commands. (See the Wolfram web site for more about [parallel commands in Mathematica](https://reference.wolfram.com/language/ParallelTools/tutorial/GettingStarted.html).) Following is an example script that works on Cori. With Cori, you can use up to 16 cores; with Edison, you can use up to 12 cores. Be sure the first line of your script points to the correct directory for the machine on which you're running.

```
#!/global/common/cori/software/mathematica/10.3.0/bin/MathematicaScript -script

# Limit Mathematica to requested resources
Unprotect[$ProcessorCount];$ProcessorCount = 4;

# calculate some Mersenne primes on one core, then on many

MersennePrimeQ[n_Integer] := PrimeQ[2^n - 1]
DistributeDefinitions[MersennePrimeQ]
slow = AbsoluteTiming[Select[Range[5000], MersennePrimeQ]]
WriteString[$Output, slow]
fast = AbsoluteTiming[Parallelize[Select[Range[5000], MersennePrimeQ]]]
WriteString[$Output, fast]
```

For Edison, the first line of this script would be

```
#!/global/common/edison/das/mathematica/10.1.0/bin/MathematicaScript -script
```

## Documentation

Extensive on-line documentation is available at the Wolfram web site.

[Mathematica Documentation](http://www.wolfram.com/products/mathematica/)

