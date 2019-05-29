# MATLAB

![MATLAB logo](./images/matlablogo.png)

[MATLAB](https://www.mathworks.com/products/matlab.html) is a
high-performance language for technical computing. It integrates
computation, visualization, and programming in an easy-to-use
environment where problems and solutions are expressed in familiar
mathematical notation. MATLAB features a family of add-on
application-specific solutions called toolboxes. Toolboxes are
comprehensive collections of MATLAB functions (M-files) that extend
the MATLAB environment to solve particular classes of problems. These
are the toolboxes installed in NERSC MATLAB, along with the number of
licenses.

* Image Processing (2)
* Neural Networks (1)
* Optimization (2)
* Distributed Computing (4)
* Signal Processing (1)
* Statistics (2)
* Compiler (1)

The name MATLAB stands for matrix laboratory. MATLAB was originally
written to provide easy access to matrix software developed by the
LINPACK and EISPACK projects. Today, MATLAB engines incorporate the
LAPACK and BLAS libraries, embedding the state of the art in software
for matrix computation.

## How to Use MATLAB on Cori

MATLAB is available at NERSC on Cori.  The number of MATLAB
licenses at NERSC is not very large (currently 16), so users should
not be running a MATLAB session when it is not being actively used. If
you use NX, it's particularly easy to think you've released your
license when you haven't, since license checkouts persist between NX
sessions.

MATLAB can be run interactively or in batch mode.  MATLAB can run on a
compute node with exclusive access to the full usable memory (about
120 GB on Cori Haswell nodes) by submitting a batch job to the regular
queue.

### Running Interactively

To run MATLAB interactively on Cori, connect with ssh -X or via NX,
and do the following:

```
salloc  -q interactive -N 1 -c 32 -C haswell -t 30:00
module load matlab
matlab
```

It is also possible to run MATLAB on a Cori login node
directly. Production computing should not be undertaken on login
nodes, however. 

!!! warn 
	For long-running or compute intensive jobs, use a batch
	script.

```
module load matlab
matlab
```

### Batch Jobs

To run one instance of MATLAB non-intearctively through a batch job,
you can use the following job script on Cori:

```slurm
#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -C haswell
#SBATCH -t 00:30:00
cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
module load matlab
srun -n 1 -c 32 matlab -nodisplay -r < myjob.m -logfile myjob.log
```

Where `myjob.m` is your MATLAB script.

### Parallelism in MATLAB

To prepare a MATLAB cluster with 32 cores (within MATLAB) do:

```
cluster = parcluster('local')
cluster.NumWorkers = 32
saveAsProfile(cluster,'cori_cluster')
pp = parpool('cori_cluster', 32)
```

#### Running MATLAB Parallel Commands

The following program illustrates how MATLAB parallel commands can be
used on Cori. NERSC's license currently limits parallel use to a
single node and the number of threads that one node supports.

```
% hello-world.m
pc = parcluster('local');
parpool(pc, 32);

spmd
  rank = labindex;
  fprintf(1,'Hello %d\n',rank);
end
```

For loop-level parallelism, MATLAB provides the parfor construct.

In order to make MATLAB work in parallel on Cori, you need to make
sure it will access the MATLAB libraries before the standard cray
libraries. You can do this by running this command for bash:

```
export LD_PRELOAD="/global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichnem.so /global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichsock.so"
```

or for csh / tcsh:

```
setenv LD_PRELOAD "/global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichnem.so /global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichsock.so"
```

However, if you have `LD_PRELOAD` set on a login node, this will
interfere with the regular environment. It's important that you only
set the `LD_PRELOAD` variable in your parallel MATLAB job script (or
interactive job).

#### Parallelism with the MATLAB Compiler

Another way to run MATLAB in parallel is to run multiple instances of
a compiled MATLAB program. By compiling, you create a stand-alone
application that doesn't need to obtain a separate license from the
NERSC license server to run. See [MATLAB Compiler](matlab_compiler)
for details.

## Documentation

Extensive [on-line documentation](http://www.mathworks.com/) is
available. You may subscribe to the MATLAB Digest, a monthly e-mail
newsletter by sending e-mail to <subscribe@mathworks.com>.
