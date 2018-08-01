Supporting data-centric science often involves the movement of data
across file systems, multi-stage analytics and visualization. Workflow
technologies can improve the productivity and efficiency of
data-centric science by orchestrating and automating these
steps. NERSC provides support for the TaskFarmer, Swift and Fireworks
tools. We also maintain other packages like Tigres that can help users
build workflows.

## TaskFarmer

[TaskFarmer](workflow/taskfarmer.md) is a utility developed at NERSC to distribute single-node tasks across
a set of compute nodes - these can be single- or multi-core tasks. TaskFarmer tracks which
tasks have completed successfully, and allows straightforward re-submission of failed or un-run jobs from a checkpoint file.

## Swift

The Swift scripting language provides a simple, compact way to write
parallel scripts that run many copies of ordinary programs
concurrently in various workflow patterns, reducing the need for
complex parallel programming or arcane scripting. Swift is very
general, and is in use in domains ranging from earth systems to
bioinformatics to molecular modeling.

```bash
module load swift
```

## Fireworks

[FireWorks](workflow/fireworks.md) is a free, open-source code for defining, managing, and
executing scientific workflows. It can be used to automate
calculations over arbitrary computing resources, including those that
have a queueing system. Some features that distinguish FireWorks are
dynamic workflows, failure-detection routines, and built-in tools and
execution modes for running high-throughput computations at large
computing centers. It uses a centralized server model, where the
server manages the workflows and workers run the jobs.
