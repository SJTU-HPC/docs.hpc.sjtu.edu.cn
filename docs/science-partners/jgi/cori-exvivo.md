# Cori ExVivo for JGI

ExVivo is a specialized system used to run JGI applications
requiring more shared memory than available on standard Cori
Genepool hardware. 

## Access

Access to Cori ExVivo is available to all JGI users as of February 6th 
2019. To use Cori ExVivo, first connect to `cori.nersc.gov`, load 
the `esslurm` module, and request a Slurm allocation. That request
command should include an `-A` argument with your project name,
`-C skylake`, and specify QoS `jgi_exvivo`, `jgi_shared`, or
`jgi_interactive`.

!!! example
	```
	--8<-- "docs/science-partners/jgi/exvivo-example.sh"
	```

* `jgi_exvivo` is intended for production use by applications and data
sets which cannot be run on Cori Genepool due to large RAM requirements.
The maximum walltime for an allocation is 7 days.
* `jgi_interactive` is intended for exploration and development. At most
4 nodes can be allocated to this QoS. The maximum wall time is 4 hours.
* `jgi_shared` is intended for jobs which require more than 128GB RAM 
  but less than 768GB. Use `-c` and `--mem=###GB` arguments in the Slurm
  invocation to request the needed resources. 

!!! note
	Multi-node allocations will not be supported on ExVivo.

## Resources

Cori ExVivo contains 20 total nodes. Each node has the following configuration:

* 2 Intel® Xeon® Gold 6140 (Skylake) processors, 36 cores total

* 1.5 TB RAM

* 1.8 TB available local disk, Solid State Drive


The user environment on ExVivo is very similar to that of a Cori login node.
Common software is available such as Cori modules, Shifter, and Anaconda.

The following filesystems are available on ExVivo: 

* User Home

* project

* projecta

* projectb

* Cori Scratch

* seqfs

* Data and Archive (read only access)

