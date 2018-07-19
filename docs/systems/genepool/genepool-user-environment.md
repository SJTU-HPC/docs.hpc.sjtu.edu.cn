## Environment on Genepool
When you log into the Genepool system you will land in your `$HOME` directory
on NERSC's "global homes" file system.  The global homes file system is
mounted across all NERSC computation systems with the exception of PDSF.
The `$HOME` directory has quota of 40GB and 1,000,000 inodes.  To customize
your environment, by setting environment variables or aliases, you will need
to modify one of the "dot" files that NERSC has created for you.  You may
NOT modify the .bashrc or .cshrc files.  These are set to read-only on NERSC
systems and specify system specific customizations.
Instead you should modify a file called .bashrc.ext or .cshrc.ext.

## Environment Variables  

- `$HOME`
points to the location of your home directory in the filesystem

- `$BSCRATCH`
points to the location of your projectb scratch directory in the filesystem

- `$SCRATCH`
points to the "best" scratch directory you have access to
on the current system. Note it is **strongly** recommended that
you use `$BSCRATCH`. `$SCRATCH` will point to `$BSCRATCH` but this
is different on other systems.

- `$NERSC_HOST`
identifies the NERSC system environment you are presently using

- `$TMPDIR`
location of current temporary space. In an ssh-interactive session TMPDIR
will point to `$SCRATCH`. In a batch-scheduled job, TMPDIR should always be
used when writing job-specific data on the compute nodes.

## Setting Up Your Work Environment with Modules
The JGI and NERSC have been collaborating to provide a large number of
bioinformatics and many other software packages on genepool.
These software are made available to you by the modules system.
Please read the general documentation on using modules at NERSC.
There are a number of default modules.  These should usually not be unloaded
unless there is a very specific need to do so
(e.g. swap PrgEnv/gnu4.6 for PrgEnv/gnu4.8 if needed).

### Notable Genepool Default Modules
- **modules**: Sets the modules environment for you
- **PrgEnv-gnu/7.1**: Programming environment module that manages the GNU gcc environment and libraries.
- **gcc/7.1.0**: Compiler that is loaded the environment. All dynamically linked libraries are built using this version of gcc.
- **nsg/1.2.0**: NERSC System Group Utilities. Provides NERSC tools to query account information (myquota) and for collaboration (collabsu)
- **OFED/4.0-2.0.0.1-Mellanox**: On Infiniband-enabled nodes, the OFED module will be loaded to support software which can make use of the high-speed interconnects
- **usg-default-modules/1.4**: Loads default modules determined by NERSC

## Common Variables Set by Modules
The modules system works by manipulating your environment.
Modules typically insert directory paths at the beginning of each of several
environment variables.  The common environment variables set by Genepool
modules:

- `$PATH`: the PATH specifies the directories where commands can be found.
- `$LD_LIBRARY_PATH`: the directories where shared libraries can be located.
- `$MANPATH`: to find manual pages.
- `$PYTHONPATH`: search paths for python packages.
- `$PERL5DIR`: search paths for perl packages.
- `$PKG_CONFIG_PATH`: the pkgconfig system enables automated discovery of libraries by autconf and automake tools.
- `<MODULENAME>_DIR`: the location of a given package.

## Tips and Best Practices
If you are considering manually encoding a path with `/global/common/` or `/usr/common/` into your software or into
you environment, please use the module instead (ie module load <package>). There are multiple paths to the software
installed in this location, and the proper way to access it depends on your current context.  

If you need to refer to a path in the module, use **[MODULENAME]_DIR** environment variable. You can see all the
settings of a module by entering: `module show <modulename>`.

Loading modules can have additional effects. The Genepool modules are
interconnected to ease the loading of dependencies. Frequently, when you load
a module, swap a module, or remove modules other modules may be loaded, swapped,
or removed.  

###  Working with Modules for Production-Level Batch Scripts
When writing a batch script which you may share with another genepool user,
it can be difficult to ensure the environment the other user will be
compatible with your batch script - usually because of differences in the
dotfile configurations and interactive-usage preferences.
For this reason, it is recommended that if you choose to load additional
modules (or unload them) in your dotfiles, that you carefully consider how
this will affect your jobs as well.  One good practice for getting
reproducible results from a batch script is to purge all modules from the
environment, and manually construct the exact environment you need:


```sh
#!/bin/bash
module purge
module load usg-default-modules
module load blast+/2.6.0

blastn ...
```

Purging the module environment and then loading the specific version of
needed modules is the recommended approach for batch scripts.
Transmitting the full environment through qsub (using the `-V`) flag is not
recommended because it is implicitly context-dependent and potentially leads
to non-reproducible calculations.

!!!warning
	module purge does not currently work on Cori and Edison. If you plan on moving your script to Cori or Edison do **not** use module purge.

### Loading Modules by Default in the dotfiles
For common tasks in an interactive environment it can be convenient to load
certain modules by default for all interactive sessions.  If this is needed,
the recommended mechanism is to embed the module commands into your
.bashrc.ext or .tcshrc.ext (depending on if you are a bash or tcsh user).  
Each NERSC system has different modules, for this reason, but your dotfiles
are evaluated by all systems.  Thus, you should check to make sure that
`$NERSC_HOST` is `genepool`, when loading genepool modules.

**bash**
```sh
## .bashrc.ext Example
if [ "$NERSC_HOST" == "genepool" ]; then
  # make user-specific changes to PATH
  export PATH="${HOME}/scripts:${PATH}"

  # then load modules
  module load blast+
fi
```

**tsch**
```sh
## .tschrc.ext Example
if ($NERSC_HOST == "genepool") then
  # make user-specific changes to PATH
  setenv PATH $HOME/scripts:$PATH
  
  # then load modules
  module load blast+
endif
```

If you alter one of the commonly manipulated environment variables in your
dotfiles, it is critical that you take extreme care.
For example, if you manually add `/jgi/tools/bin` to your `PATH` - and have
the jgitools module loaded, the evaluation order of the `PATH` will likely
be incorrect and you may experience unexpected side effects.
It is recommended to make any manual modifications to `PATH`,
`LD_LIBRARY_PATH`, and others earlier in the dotfiles than the module commands.

### Using Modules in Cron Jobs  
The cron environment on genepool does not have a complete environment setup.
In particular important environment variables like `$SCRATCH`, `$HOME`,
`$BSCRATCH`, `$NERSC_HOST` may not be setup.
Also, modules will not work.  To get a proper environment for a cron job,
you'll need to start a new login-style shell for your process to work in.
For a simple job this can be done like:

```sh
# crontab
07 04 * * * bash -l -c "module load python; python /path/to/myScript.py"
```  
If you need a more extensive environment setup, you can simply put the entire
cronjob into a script, and call the script from your crontab.

```sh
*** script
#!/bin/bash -l
module load python
module load hdf5
…

*** crontab entry
07 04 * * * /path/to/myScript
```

The key with both of these methods is the `bash -l` which is ensuring that a
new environment is initialized for the shell which will be complete (including
modules).

# Working With Modules Within Perl and Python
It can often be convenient to work with the modules system within perl or python
scripts. In this section, we present ways to do this.

## Using Modules Within Python
The `EnvironmentModules` python package gives access to the module system
from within python.  The `EnvironmentModules` python package has a single
function: module.  Using this function you can provide the same arguments
you would to `module` on the command line.  The `module()` function accepts a
list of arguments, like `'load','<modulename>'`; or `'unload','<modulename>'`.

```python
import EnvironmentModules as EnvMod
EnvMod.module(['load', 'blast+'])
```  
It is important to understand that this is most effective for scripts which
execute other code (e.g. from the subprocess package of python), and not
necessarily for loading additional packages for python to use.  This is
because the python process is already running and changing its environment
won't necessarily give expected results.  For example, changes to `PYTHONPATH`
and `LD_LIBRARY_PATH` are not immediately accepted.  `LD_LIBRARY_PATH` is
only evaluated at process start-up time, and won't be re-evaluated later.  
Thus if you load any python packages which rely on dynamically linked C-code,
you should load those modules before python (oracle_client, for example).

### Problems with LD_LIBRARY_PATH
**Wrong Way**
```python
EnvMod.module(['load', 'oracle_client'])
import cx_Oracle
Traceback (most recent last call):
    File "", line, in
    ImportError: libcIntsh.so.11.1: cannot open shared object file: No such file or directory
```

**Right Way**
```sh
module load oracle_client
python
Python 2.7.14 [Anaconda custom (64-bit)] (default, Mar 27 2018, 17:39:31)
[GCC 7.2.0] on linux2
Type "help", "copyright" or "license" for more information
»> import cx_Oracle
```

This is happening because the cx_Oracle python package relies on the
dynamically linked library libclntsh.so.11.1, which is found by the
operating system by setting the correct path in `LD_LIBRARY_PATH`. Since
`LD_LIBRARY_PATH` is only evaluated when python starts, the oracle_client
modulefile (which sets `LD_LIBRARY_PATH` for oracle), needs to be loaded
before python is started.  For some applications, it may be easier to
bootstrap the scripting session with a scriptEnv.


### Workaround for PYTHONPATH
The `PYTHONPATH` environment variable is set to sys.path when python starts.
There is an opportunity to adjust sys.path and still take advantage to
changes to `PYTHONPATH` if you need this functionality:

```python
import os
import sys
import EnvironmentModules as EnvMod
for x in os.getenv("PYTHONPATH").split(":"):
  if x notr in sys.path:
    sys.path.append(x)

from Bio import SeqIO
```

### Using Modules within Perl
The `EnvironmentModules` perl package gives access to the module system from
within perl.  Just like the python package, it has a single function:
module. Using this function you can provide the same arguments you would
to `module` on the command line.  The `module()` function accepts a single
string or array representing the arguments you would pass to the module
command line tool.  Note that `EnvironmentModules.pm` is only installed in
the modules version of perl, and thus you need to load the perl module
before you can access `EnvironmentModules.pm`.

```perl
#!/usr/bin/env perl
use EnvironmentModules;
module("load blast+");
```  

Please note that the similar to python, loading modules which manipulate
`LD_LIBRARY_PATH` or `PERL5DIR` will not work as expected.  It is generally
recommended to load the modules before entering into perl or python instances.
Another popular method, loading the module in system() immediately before
running an executable should always be avoided.  This is not portable, and
will only work if your users are using the bash shell:

```perl
#!/usr/bin/env perl

### DO NOT DO THIS!
system("module load blast+"; blastn …");
```
 When you call `system()`, perl forks it as `/bin/sh -c '<your command'>`.  
 `/bin/sh` does not get a new module environment loaded, so, the instance of
 `/bin/sh` will be relying on the shell operating perl to get the module
 functionality.  It turns out that the module functionality provided by bash
 is correct for `/bin/sh`, but not the module functionality provided in
 tcsh or csh.  If you do the above, your code will only work for other
 bash users.  Instead, you should either load modules before running perl
 (a wrapper script -- preferred), or use the EnvironmentModules mechanism
 shown above.


# Collaboration Accounts On Genepool

## Overview
The production computing environment on the genepool system has been set up
to allow, upon request, collaboration accounts to be created.  The purpose
of these collaboration accounts is to allow collections of users to equally
access and manipulate files and jobs run by the collaboration user.

## Requesting and maintaining Collaboration Accounts
Genepool PIs, PI proxies, and JGI group leads can request collaboration
accounts.  Please file a ticket to request a collaboration account by visiting
https://help.nersc.gov to file a ticket.  Furthermore, only genepool PIs,
PI proxies and JGI group leads can request changes in membership to
collaboration accounts.  Please file a service ticket to change collaboration
account membership or host settings.

## Using Collaboration Account
Collaboration accounts on genepool will allow permitted users on specific
hosts to switch users.  To switch users on genepool, first ssh to your group
gpint system and then run the `collabsu` command, and finally enter your NIM
password to gain access to the collaboration account.  `collabsu` is a
replacement for sudo which allows user-level secured switching on the diverse
and complex genepool environment.

```sh
mamelara@denovo:~$ ssh gpint
…
mamelara@gpint13:~$ collabsu annotrub
[sudo] password for mamelara:
annotrub@gpint13:~$
```

## Restrictions on Collaboration Accounts
Collaboration accounts are a special class of account which do not allow
direct password access.  If you have a legacy collaboration account which
does allow password access, please expect us to contact you about converting
it to a modernized and secure collaboration account.  If you do have password
access to a legacy collaboration account, please remember that it not
permissible for users to share passwords for NERSC accounts, nor is it
permissible to hold multiple NERSC accounts.

The policy for running `collabsu` is set by system, so you will be permitted
to run `collabsu` on any genepool/gpweb system, but not on other NERSC
platforms.

Collaboration accounts are afforded the same privileges as other NERSC-user
accounts, including the same quota limits on /global/homes and scratch
directories. It is important to be careful that the collaboration account
does not exceed quotas.  Maintaining coordination between multiple users can
be challenging, so ensure you are communicating regularly with your co-users
of the collaboration account.
