# WIEN2k 

[WIEN2k](http://susi.theochem.tuwien.ac.at/) 
performs electronic structure calculations of solids using density functional theory (DFT). It is based on the full-potential (linearized) augmented plane-wave ((L)APW) + local orbitals (lo) method, one of the most accurate schemes for band structure calculations.

Wien2k is available on Cori. It is important for users to follow the instructions posted here to use Wien2k on NERSC systems.

## How to access WIEN2k
NERSC uses [modules](../../environment/#nersc-modules-environment) to manage access to software. 
To use the default version of WIEN2k, include the following line:
```shell
cori$ module load wien2k
```
in your ~/.bashrc.ext or ~/.cshrc.ext files. To see all the available modules, do

```shell
cori$ module avail wien2k
```
## How to run WIEN2K 
To run Wien2k jobs on Cori, you need to do the following:

1) Loading the wien2k module in your shell startup file, ~/.bashrc.ext or ~/.cshrc.ext file for Cori, depending on your default shell (~/.cshrc.ext for csh, ~/.bashrc.ext for bash).

```shell
cori$ module load wien2k
```
If you run Wien2k jobs on a single node only, then you do not have to have to load the wien2k module in the shell startup files. 

2) Using a job script similar to the sample job scripts below to submit batch jobs

Sample job script to run a k point + MPI parallel job on Cori Haswell nodes
!!! example "Cori Haswell"
    ```slurm
    --8<-- "docs/applications/wien2k/cori-hsw-kpoints.sh"
    ```

This job script requests 4 nodes. Where the script gen.machines is a NERSC provided script that generates the .machines file for Wien2k parallel execution from the SLURM_JOB_NODELIST environment. ($SLURM_JOB_NODELIST is the nodelist provided allocated to your job). In this example, each kpoint group will use 8 cores to run its mpi parallel execution. Here is more info about how to use this script to generate a desired .machines file.

Sample job script to run a k point parallel job on Cori

!!! example "Cori Haswell"
    ```slurm
    --8<-- "docs/applications/wien2k/cori-hsw-kpoints-mpi.sh"
    ```

## Using w2web 

!!! Note 
	W2web is currently disabled on Cori due to a security issue. We will let you know when it is re-enabled. 

Wien2k on Cori was built with the graphical user interface w2web enabled. To use w2web, you must use a local ssh tunnel for security concern. You should not point a web browser to directly to Cori (Wien2k manual may instruct you to do so), instead you should use a secure local tunnel to forward a given port to it, but it does allow a given local port to be forwarded through a secure local tunnel.

You can use ssh as follows:

```shell
ssh -L 4650:localhost:4660 cori.nersc.gov  
```
After you login, you should load the wien2k module, and then run the w2web application. It will ask you to pick a port number: in this example the port number is 4660. After pointing your web browser to `http://localhost:4650`, you should be able to see w2web's startup screen.

If you use windows and the F_secure or putty shell client to access Cori, then specify a local tunnel as follows.

Open the F-Secure shell client, and choose Edit-->settings.  This will pop up a settings window, in which you should click add. Another pop-up window will then appear, requesting that you define your local tunnel.  Enter

4650 in the source port field
4660 in the destination field
cori.nersc.gov in the destination host field
Using local tunnels with putty is similar, except that for the destination host field you enter cori.nersc.gov:4660 (there is no separate field for the destination port as with F-secure).

Note: the port numbers 4660 and 4650 used here are just examples, you can choose any port number larger than 1024 and less than 65535.


## Documentation

WIEN2K was developed at the Institute for Materials Chemistry at the Technical University, Vienna. 
The [Wien2k Textbooks](http://www.wien2k.at/reg_user/textbooks/) website has a detailed description of the electronic structure methods available in Wien2k, and a set of technical notes. The [Wien2k Users Manual](http://www.wien2k.at/reg_user/textbooks/usersguide.pdf) is particularly useful.

