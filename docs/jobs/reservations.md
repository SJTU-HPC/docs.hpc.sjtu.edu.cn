# Reservations

Users can request a scheduled reservation of machine resources if
their jobs have special needs that cannot be accommodated through the
regular batch system. A reservation brings some portion of the machine
to a specific user or project for an agreed upon duration. Typically
this is used for interactive debugging at scale or real time
processing linked to some experiment or event.

!!! failure "Note"
	It is not intended to be used to guarantee fast throughput
	for production runs.

## Charging

For normal batch jobs, charging against a project's allocation is done
on a per job basis. For scheduled reservations the entire block of
reserved time is charged regardless of the number of nodes used or
time spent running jobs.

## Requesting a reservation

To reserve compute nodes, a request must be sent in with *at least 1 week*
notice.  Please ask for the least amount of resources you need
and try to schedule reservations so as to minimize impact on other
users. It is also recommended that reservations be scheduled to start
during NERSC business hours to ensure availability of staff in case
any issues arise.

[Reservation Request Form](https://nersc.service-now.com/catalog_home.do?sysparm_view=catalog_default)

## Cancellations

Cancellation of a reservation must be done with a minimum of 4 days
notice. If you do not receive a confirmation that your
cancellation was received and it is less than 4 days until your
start time you *must* contact NERSC operations via 1-800-666-3772 (or
1-510-486-8600) menu option 1 to confirm.

## Using a reservation

Once your reservation request is approved and a reservation is placed
on the system use the `--reservation` option:

```
nersc$ sbatch --reservation=<reservation_name>
```

```
nersc$ salloc --reservation=<reservation_name>
```

or add `#SBATCH --reservation=<reservation_name>` to your job script.

!!! note
	It is possible to submit jobs to a reservation once it is
	*created* - jobs will start immediately when the reservation is
	available.

### KNL mode changes

!!! warning
	KNL reboots can take up to 1 hour.

KNL nodes are configured as quad,cache by default. If you have
requested a reservation with a different mode you will be responsible
for rebooting the nodes into the proper mode and should account for
this in your request.

!!! note
	Additional demonstration of need may be required for large scale
	reservations requesting reboots as system administrators are
	needed to assist in the reboot.

It is recommended to submit a "test" job at the start of your
reservation to reboot the nodes.

```shell
	cori$ sbatch -C knl,quad,flat --nodes=<size_of_reservation> --qos=regular --reservation=<reservation_name> --wrap="hostname"
```

!!! danger
	If you forget to put the correct `-C` option on **all** of your jobs
	**you may lose 2 hours of your reservation** to reboots.

## Ending a reservation

All running jobs under a reservation will be terminated when the
reservation ends. If you complete the planned computations before the
reservation ends, please call NERSC operations at 1-800-666-3772 (or
1-510-486-8600) menu option 1 to cancel the reservation.
