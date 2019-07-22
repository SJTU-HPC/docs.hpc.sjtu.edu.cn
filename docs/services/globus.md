# Globus
## Overview
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

## Availability
Globus is available as a free service that any user can access.  You
can log into [Globus web interface](https://globus.org) with your
NERSC credentials (by selecting NERSC in the drop down menu of
supported identity providers) or using many of the other supported
providers listed that you can authenticate against. NERSC maintains
several Globus endpoints that can be activated for individual use by
any NERSC user. The list of endpoints are provided in the table below.

| Endpoint Name | Description |Recommended Use
| --- | --- | ---
| NERSC DTN | Multi-node, high performance transfer system with access to all NERSC Global File systems (NGF) as well as Cori Scratch |  Almost all data transfers needs into & out of NERSC
| NERSC HPSS | Single node system connected directly to the NERSC HPSS tape archive | Remote transfers into & out of HPSS
| NERSC Cori | Originally a dual-node system needed for accessing the Cori scratch file system. The endpoint is the same as NERSC DTN |  Use NERSC DTN instead
| NERSC DTN-JGI | Single node system that was used to access JGI-specific file systems, which are now connected to the NERSC DTN servers. | Use NERSC DTN instead
| NERSC SHARE | Single node system with read-only access to the project file system | Shared Globus endpoint

!!! warning 
    Be aware that Globus will follow hard links and copy the data to
    the target directory. Depending on the way the directory or file
    is transferred, it will also follow soft links. You can find
    details here:
    https://docs.globus.org/faq/transfer-sharing/#how_does_globus_handle_symlinks

!!! tip
    **To transfer files into/out of your laptop or desktop
    computer, you can install a [Globus connect (personal)
    server](https://www.globus.org/globus-connect) to configure an
    endpoint on your personal device.**

## Globus Sharing
Data can be shared at NERSC using [Globus
Sharing](https://www.globus.org/data-sharing). Currently shared
endpoints are read-only, no writing is allowed. To share data, create
a "gsharing" directory in your repo's project directory. Then [open a
ticket](https://help.nersc.gov/) and let us know the directory path
you'd like to share (we currently have to manually update the
configuration file to allow directories to be shared). Once you hear
back from us, go to the [Globus endpoint web
page](https://app.globus.org/endpoints) and search for the "NERSC
SHARE" endpoint. Click on the "Shares" tab and select "Add a Shared
Endpoint". This will bring you to a screen where you can give your
shared endpoint a name and fill in the path you'd like to
share. Currently sharing is limited to
/global/project/projectdirs/<your_repo_name\>/gsharing and
subdirectories (and the [dna file
system](../../science-partners/jgi/filesystems/#dna-data-n-archive)
for JGI users). Once you click "Create Share" it will take you to
another screen where you can share this endpoint with specific Globus
users or with all Globus users (these users do **not** have to be
NERSC users). You can also make other Globus users administrators,
which will mean that they will have the power to add or remove other
users from the shared endpoint.

## Command Line Globus Transfers
### Globus SDK
Globus provides a [python based
SDK](https://globus-sdk-python.readthedocs.io/en/stable/) (Software
Development Kit) for doing data transfers, managing endpoints,
etc. You can access this at NERSC by loading the globus-sdk module
(`module load globus-sdk`). NERSC has written a helper script,
`transfer_files.py` to help with command line data transfers. You can
use this script for internal transfers between NERSC file systems or
for external transfers between two Globus endpoints provided you have
the endpoint's UUID (which can be looked up on [Globus' Endpoint
Page](https://app.globus.org/endpoints)). Usage is:

```
cori04> transfer_files.py --help
usage: transfer_files.py [-h] -s SOURCE -t TARGET -d OUT_DIR -i INFILES

Globus transfer helper

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        source endpoint UUID
  -t TARGET, --target TARGET
                        target endpoint UUID
  -d OUT_DIR, --directory OUT_DIR
                        target endpoint output directory
  -i INFILES, --infiles INFILES
                        file containing list of full path of input files
```

This script also understands "hpss" (the NERSC HPSS Endpoint), "dtn"
(the NERSC DTN Endpoint), or "cori" (the NERSC Cori Endpoint), so you
won't need to look up the UUIDs for those endpoints.

!!! warning
    If you are using Globus to read a large number (>100) of files
    from NERSC's Archive, please use this script. It will ensure that
    the files are read off in in tape order.

When you run this script, it will output a web page address. Paste
this page into a browser and it will take you to [a
page](globus_native_app_grant.png) where you can grant the native
globus app permission to make transfers on your behalf (you may get a
prompt to log into globus first if you don't already have an active
session). Once you click "Allow" it will generate a one-time
code. Paste this code back into the terminal at the prompt and push
return. It will use this code to get a one-time authentication and
generate a globus transfer on your behalf. It will print out the
transfer ID which you cna use to check on the transfer on the [Globus
web page](https://www.globus.org) (or with the Globus SDK).


#### Example
##### Transfer files from [NERSC's HPSS Archive](../filesystems/archive.md) to another location
First, generate a list of files you'd like to transfer:
```
hsi -q ls -1 -R name_of_hpss_directory_or_file 2> gtransfer_list.txt
```
Then invoke the transfer script
```
cori04> transfer_files.py -s hpss -t <target_endpoint_UUID> -d /your/destination/path -i gtransfer_list.txt
Please go to this URL and login: https://auth.globus.org/v2/oauth2/authorize?client_id=<super_mega_long_string_here>
Please enter the code you get after login here: <snipped>
Transfer ID is b'XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX'
```
## Troubleshooting
### Connection Errors Between NERSC and other endpoint
If you are getting errors that you cannot connect to the NERSC
endpoint after you've activated it, please check with your system
administrator that they are not blocking the IP of the NERSC host (you
can find this information in the error message of the Activity pane on
the globus web page). If they are not, please open a ticket with the
IP address of the other endpoint and we will investigate further.
### Trouble Activating a NERSC Endpoint
If you are having trouble activating a NERSC endpoint, please try
logging into [NIM](https://nim.nersc.gov) to clear your authentication
failures. If that still doesn't fix the issue, please open a ticket
with the error and we'll investigate further.