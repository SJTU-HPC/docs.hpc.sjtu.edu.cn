# Overview

[Globus](https://globus.org) is the recommended way to move
significant amounts of data between NERSC and other sites. Globus
addresses many of the common challenges faced by researchers in
moving, sharing, and archiving large volumes of data. With Globus, you
hand-off data movement tasks to a hosted service that manages the
entire operation, monitoring performance and errors, retrying failed
transfers, correcting problems automatically whenever possible, and
reporting status to keep you informed while you focus on your
research. Visit [Globus.org](https://globus.org) for documentation on
its easy to use web interface and its versatile REST/API for building
scripted tasks and operations.

# Availability

Globus is available as a free service that any user can access.
You can log into Globus web interface with your NERSC
credentials (by selecting NERSC in the drop down menu of supported
identity providers) or using many of the other supported providers
listed that you can authenticate against. NERSC maintains several
Globus endpoints that can be activated for individual use by any NERSC
user. The list of endpoints are provided in the table below.

| Endpoint Name | Description |Recommended Use
| --- | --- | ---
| NERSC DTN | Multi-node, high performance transfer system with access to all NERSC Global File systems (NGF) as well as Cori Scratch |  Almost all data transfers needs into & out of NERSC
| NERSC HPSS | Single node system connected directly to the NERSC HPSS  tape archive | Remote transfers into & out of HPSS
| NERSC Edison | Single node system connected to NGF and uniquely to  the Edison scratch file system | Only recommended for access to  Edison scratch
| NERSC PDSF | Single node system connected to NGF | Use NERSC DTN  instead
| NERSC Cori | Originally a dual-node system needed for accessing the Cori scratch file system.  The endpoint is the same as NERSC DTN |  Use NERSC DTN instead
| NERSC DTN-JGI | Single node system that was used to access JGI-specific file systems, which are now connected to the NERSC DTN servers. | Use NERSC DTN instead

!!! tip
    **To transfer files into/out of your laptop or desktop
    computer, you can install a [Globus connect (personal)
    server](https://www.globus.org/globus-connect) to configure an
    endpoint on your personal device.**

# Troubleshooting

## Connection Errors Between NERSC and other endpoint
If you are getting errors that you cannot connect to the NERSC
endpoint after you've activated it, please check with your system
administrator that they are not blocking the IP of the NERSC host (you
can find this information in the error message of the Activity pane on
the globus web page). If they are not, please open a ticket with the
IP address of the other endpoint and we will investigate further.

## Trouble Activating a NERSC Endpoint
If you are having trouble activating a NERSC endpoint, please try
logging into [NIM](https://nim.nersc.gov) to clear your authentication
failures. If that still doesn't fix the issue, please open a ticket
with the error and we'll investigate further.
