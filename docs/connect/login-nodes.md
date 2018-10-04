# Login Nodes

Opening an [SSH connection](./ssh.md) to NERSC systems results in a
connection to a login node. Systems such as Cori and Edison have
multiple login nodes which sit behind a load balancer. New connections
will be assigned a random node. If an account has recently connected
the load balancer will attempt to connect to the same login node as
the previous connection.

## Usage

!!! warning "Appropriate Use"
	Do not run compute or memory intensive applications on login
	nodes. These nodes are a shared resource. NERSC may terminate
	processes which are having negative impacts on other users or the
	systems.

* compile codes (limit to e.g. `make -j 8`)
* edit files
* submit [jobs](/jobs/index.md)

Some workflows require interactive use of applications such as IDL,
MATLAB, NCL, python, and ROOT. For **small** datasets and **short**
runtimes it is acceptable to run these on login nodes. For extended
runtimes or large datasets these should be run in the batch queues.

!!! tip
	An[interactive qos](/jobs/examples/#interactive) is available
	on Cori for compute and memory intensive interactive work.

### Data transfers

If you need to do a large number of data transfers use a dedicated
xfer queue.

### Guidelines

*  do not use more than 50% of the cores (command: `lstopo -p --only
   core | wc -l`)
*  do not use more than 25% of the memory (command: `free -m`)
*  avoid the `watch` command
*  avoid long running commands
