# MATLAB

![MATLAB logo](./images/matlablogo.png)<br/> 
[MATLAB](https://www.mathworks.com/products/matlab.html) is a high-performance language for technical computing. It integrates computation, visualization, and programming in an easy-to-use environment where problems and solutions are expressed in familiar mathematical notation. MATLAB features a family of add-on application-specific solutions called toolboxes. Toolboxes are comprehensive collections of MATLAB functions (M-files) that extend the MATLAB environment to solve particular classes of problems. These are the toolboxes installed in NERSC MATLAB, along with the number of licenses.

* Image Processing (2)
* Neural Networks (1)
* Optimization (2)
* Distributed Computing (4)
* Signal Processing (1)
* Statistics (2)
* Compiler (1)

The name MATLAB stands for matrix laboratory. MATLAB was originally written to provide easy access to matrix software developed by the LINPACK and EISPACK projects. Today, MATLAB engines incorporate the LAPACK and BLAS libraries, embedding the state of the art in software for matrix computation.

## How to Use MATLAB on Edison and Cori

MATLAB is available at NERSC on Edison and Cori.  The number of MATLAB licenses at NERSC is not very large (currently 16), so users should not be running a MATLAB session when it is not being actively used. If you use NX, it's particularly easy to think you've released your license when you haven't, since license checkouts persist between NX sessions.

MATLAB can be run interactively or in batch mode.  MATLAB can run on a compute node with exclusive access to the full usable memory (about 62 GB on Edison nodes and 120 GB on Cori Phase 1 nodes) by submitting a batch job to the regular queue.

### Running Interactively

To run MATLAB interactively on Edison, connect with ssh -X or via NX, and do the following:

```
salloc -p regular -N 1 -c 24 -t 30:00
module load matlab
matlab
```

It is also possible to run MATLAB on an Edison login node directly:

```
module load matlab
matlab
```

### Batch Jobs

To run one instance of MATLAB non-intearctively through a batch job, you can use the following job script on Edison:

```
#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -t 00:30:00
cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
module load matlab
matlab -nodisplay -r < myjob -logfile myjob.log
```
Where `myjob.m` is your MATLAB script. 

### Parallelism in MATLAB

To prepare a MATLAB cluster with 24 cores (within MATLAB) do:

```
cluster = parcluster('local')
cluster.NumWorkers = 24
saveAsProfile(cluster,'edison_cluster')
pp = parpool('edison_cluster', 24)
```

#### Running MATLAB Parallel Commands

The following program illustrates how MATLAB parallel commands can be used on Cori. NERSC's license currently limits parallel use to a single node and the number of threads that one node supports.

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

In order to make MATLAB work in parallel on Edison, you need to make sure it will access the MATLAB libraries before the standard cray libraries. You can do this by running this command for bash:

```
export LD_PRELOAD="/global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichnem.so /global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichsock.so"
```

or for csh / tcsh:

```
setenv LD_PRELOAD "/global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichnem.so /global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichsock.so"
```

However, if you have LD_PRELOAD set on a login node, this will interfere with the regular Edison and Cori environment. It's important that you only set the LD_PRELOAD variable in your parallel MATLAB job script (or interactive job).

#### Parallelism with the MATLAB Compiler

Another way to run MATLAB in parallel is to run multiple instances of a compiled MATLAB program. By compiling, you create a stand-alone application that doesn't need to obtain a separate license from the NERSC license server to run. The MathWorks web site has details about using the [MATLAB Compiler](https://www.mathworks.com/products/compiler.html).

## Documentation

Extensive [on-line documentation](http://www.mathworks.com/) is available. You may subscribe to the MATLAB Digest, a monthly e-mail newsletter by sending e-mail to subscribe@mathworks.com.

