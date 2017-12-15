# NERSC User Environment

## Home Directories, Shells and Dotfiles 

All NERSC systems use global home directories, which are are pre-populated 
with shell initialization files (also known as dotfiles) for all available 
shells. NERSC fully supports `bash`, `csh`, and `tcsh` as login shells. Other shells 
(`ksh`, `sh`, and `zsh`) are also available. The default shell at NERSC is bash.  

## Dotfiles 
The "standard" dotfiles are symbolic links to read-only files that NERSC 
controls. For each standard dotfile, there is a user-writeable ".ext" file.
For example, C-shell users are generall concerned with the files .login and
.cshrc, which are read-only NERSC. These users should put their customizations 
in .login.ext and .cshrc.ext.  

Users may have certain customizations that are appropriate for one NERSC platform,
but not for others. This can be accomplished by testing the value of the
environment variable `$NERSC_HOST`. For example, on Edison and Cori the default 
programming environment is Intel (PrgEnv-Intel). A C-shell user who wants to
use the `GNU` programming environment should include the following module 
command in their `.cshrc.ext` file:  

### Edison version  
```bash
if ($NERSC_HOST == "edison") then
  module swap PrgEnv-intel PrgEnv-gnu
endif
```

### Cori version
```bash
if ($NERSC_HOST == "cori") then
  module swap PrgEnv-intel PrgEnv-gnu
endif
```

### Fixing Dotfiles  
Ocassionally, a user will accidentally delete the symbolic links to the standard
dotfiles, or otherwise damange the dotfiles to the point that it becomes
difficult to do anything. In this case, the user should runt he command
`fixdots`. This will recreate the original dotfile configuration, after first
saving the current configuration in the directory `$HOME/KeepDots.timestamp`
is a string that includes the current date and time. After running `fixdots`,
the user should carefully incorporate the saved customizations into the 
newly-created .ext files.  

### Changing Default Login Shell
Use **NIM** to change your default login shell. Login, then select **Change Shell**
from the **Actions** pull-down menu.  

## NERSC Modules Environment
NERSC uses the module utility to manage nearly all software. There are two
huge advantages of the module approach:  
1. NERSC can provide many different versions and/or installations of a single
software pacakge on a given machine, including a default version as well as 
several older and newer version.

2. Users can easily switch to different versions or installations without
having to explicitly specify different paths. With modules, the `MANPATH` and
related environment variables are automatically managed. 

### Module Command 
#### module help  
To get a usage list of module options type the following (listing is abbreviated):  
```bash
  module help

  Available Commands and Usage:
     
     + add|unload      modulefile [modulefile …]
     + rm|unload       modulefile [modulefile …]
     + switch          modulefile1 modulefile2
     + display         modulefile [modulefile …]
     + avail           path [path]
     + list            
     + help            modulefile [modulefile …]
```

#### module list
```bash
   module list
```

#### module avail
```bash
   module avail
```
