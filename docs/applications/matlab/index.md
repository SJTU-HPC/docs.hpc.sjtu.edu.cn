# MATLAB

![Matlab logo](./images/matlablogo.png)<br/> 
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

Matlab is available at NERSC on Edison and Cori.  The number of Matlab licenses at NERSC is not very large (currently 16), so users should not be running a Matlab session when it is not being actively used. If you use NX, it's particularly easy to think you've released your license when you haven't, since license checkouts persist between NX sessions.

Matlab can be run interactively or in batch mode.  Matlab can run on a compute node with exclusive access to the full usable memory (about 62 GB on Edison nodes and 120 GB on Cori Phase 1 nodes) by submitting a batch job to the regular queue.

### Running Interactively

To run Matlab interactively on Edison, connect with ssh -X or via NX, and do the following:

```
salloc -p regular -N 1 -c 24 -t 30:00
module load matlab
matlab
```

It is also possible to run Matlab on an Edison login node directly:

```
module load matlab
matlab
```

### Batch Jobs

To run one instance of matlab non-intearctively through a batch job, you can use the following job script on Edison:

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
Where `myjob.m` is your matlab script. 

### Parallelism in MATLAB

To prepare a MATLAB cluster with 24 cores (within matlab) do:

```
cluster = parcluster('local')
cluster.NumWorkers = 24
saveAsProfile(cluster,'edison_cluster')
pp = parpool('edison_cluster', 24)
```

#### Running MATLAB Parallel Commnands

The following program illustrates how matlab parallel commands can be used on Cori. NERSC's license currently limits parallel use to a single node and the number of threads that one node supports.

```
% hello-world.m
pc = parcluster('local');
parpool(pc, 32);
 
spmd
  rank = labindex;
  fprintf(1,'Hello %d\n',rank);
end
```

For loop-level parallelism, Matlab provides the parfor construct.

In order to make matlab work in parallel on Edison, you need to make sure it will access the Matlab libraries before the standard cray libraries. You can do this by running this command for bash:

```
export LD_PRELOAD="/global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichnem.so /global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichsock.so"
```

or for csh / tcsh:

```
setenv LD_PRELOAD "/global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichnem.so /global/common/sw/cray/cnl6/ivybridge/matlab/R2016b/bin/glnxa64/libmpichsock.so"
```

However, if you have LD_PRELOAD set on a login node, this will interfere with the regular Edison and Cori environment. It's important that you only set the LD_PRELOAD variable in your parallel Matlab job script (or interactive job).

#### Parallelism with the MatLab Compiler

Another way to run MatLab in parallel is to run multiple instances of a compiled MatLab program. By compiling, you create a stand-alone application that doesn't need to obtain a separate license from the NERSC license server to run. The MathWorks web site has details about using the [MatLab Compiler](https://www.mathworks.com/products/compiler.html).

## Documentation

Extensive [on-line documentation](http://www.mathworks.com/) is available. You may subscribe to the MATLAB Digest, a monthly e-mail newsletter by sending e-mail to subscribe@mathworks.com.

