# NX / NoMachine

NX (aka NoMachine) is a computer program that handles remote X Window
System connections which offers several benefits over traditional X
forwarding.

## Benefits

### Speed

NX can greatly improve the performance of X Windows, allowing users
with slow, high latency connections (e.g. on cell phone network,
traveling in Africa) to use complex X Windows programs (such as
rotating a plot in Matlab).

### Session

NX provides functionality that allow a user to disconnect from the session
and reconnect to it at a later time while keeping the state of all
running applications inside the session.

### Desktop

NX gives users a virtual KDE desktop that runs at NERSC. You can
customize the desktop according to your work requirement.

## Installing

### Download latest NX Client
#### Windows XP/Windows 7/Windows 8 ([Download](https://portal.nersc.gov/project/mpccc/nx/nomachine-enterprise-client_6.0.80_1.exe))
#### Mac OS X 10.5 or later on an Intel Mac ([Download](https://portal.nersc.gov/project/mpccc/nx/nomachine-enterprise-client_6.0.66_2.dmg))
#### DEB Package for Debian/Ubuntu: ([Download](https://portal.nersc.gov/project/mpccc/nx/nomachine-enterprise-client_6.0.66_2_amd64.deb))
#### RPM Package for Centos/Redhat/Fedora ([Download](https://portal.nersc.gov/project/mpccc/nx/nomachine-enterprise-client_6.0.66_2_x86_64.rpm))

To configure open the NX player and click on the box in the
upper left that says "New". Use "SSH" for protocol,
"nxcloud01.nersc.gov" for host, "password" for authentication,
"Don't use a proxy" for proxy. Alternatively you can [download
the configuration
file](https://portal.nersc.gov/project/mpccc/nx/Connection_to_NERSC_NX_service.nxs.gz)
(you'll need to unzip it and open it with the NX Client).

## Connecting

You can connect to NX via the NX Client or by pointing your browser
at [nxcloud01](https://nxcloud01.nersc.gov).

!!! note
    The browser interface is still experimental and can sometimes
    be slower than connecting via the client.

### Connecting with MFA

Add your one-time password to the "Password" field after your NERSC
password (with no spaces between them).

## Troubleshooting

### Connecting to NX

If you are having trouble connecting to NX, please can try these steps:

1. Log into [NIM](https://nim.nersc.gov) to clear any login
   failures. Access to NX uses your NERSC user name and password. If
   your password is mistyped five times, NERSC will lock you out of
   their systems. Logging into NIM will automatically clear these
   failures. This will also let you know if your password is expired
   (which would prevent you from accessing NX, among many other
   things).
2. Re-download
   the
   [NX configuration file](https://portal.nersc.gov/project/mpccc/nx/Connection_to_NERSC_NX_service.nxs.gz). NX
   will often "update" the configuration file to try to save your
   settings and sometimes this file can get some bad settings. You
   must have the new NX player AND the new configuration file to
   connect to the NX service.
3. Try to ssh directly to the NX server. You can do this with the
   command `ssh <nersc_username>@nxcloud01.nersc.gov` and your NERSC
   user name and password. If your access to the NX server is blocked
   by a local firewall or something else and you can't connect via
   ssh, you will also not be able to connect with the NX client.

If you've tried these steps and still cannot connect, please open a
help ticket. In this ticket, please include the following information:

1. The type of system you're trying to connect from (i.e. Mac,
   Windows, Linux, etc.).
1. A screen capture of the error you get (if possible).
1. A tarball of the NX logs. You can find instructions for how to
   bundle your NX logs on
   the [NoMachine website](https://www.nomachine.com/DT07M00098).

### Configuring the NX Environment

#### Font size is too big or too small

To change the font size inside your terminal: In the menu of Konsole
Application, choose "Settings"->"Manage Profiles", then click "Edit
Profile...", now you can change the font size in the "Appearance" tab,
after changing, click "OK" until you are back to the terminal. Now
every new terminal window you open will have the new font size.

To change the font size of your menu bars/window titles: Right click
on an empty desktop then choose "Konsole", inside the Konsole, type
"kcmshell4 fonts". Then you have a dialog box to change your font
size.

#### Resizing the NX screen

With the latest NX Player (5.0.63 or later), the most efficient way is
to enable "Remote Resize" in the NX menu:

1. Connect to NX
1. From the desktop, bring up the NX player menu with a hotkey: Mac:
   Ctrl+Option+0, Windows: Ctrl+Alt+0, Linux: Ctrl+Alt+0
1. Choose the "Display" submenu, then toggle the "Remote Resize"
   button. You can also choose "Change Settings" to manually change
   the resolution.

#### Emacs complains about missing fonts and shows white blocks

This is due to a problem with font server. Please use the following
command instead: `emacs -font 7x14`

### Keypairs and NX

The NX server acts as a gateway to all other NERSC systems. Users
access other NERSC systems via SSH with their NERSC user name and
password. The global home directories are not mounted on the NX
servers, so if you want to use SSH keys on NX, you will need to
generate a separate keypair.

#### Generating a Keypair

SSH Agent requires a keypair to function. The first time you click an
item on the "NERSC Systems" menu the keypair is created and
installed. You need to provide a password to encrypt your private
key. This password can be different from your NIM password. You can
generate a keypair by selecting the "(Re)generate Key Pair" menu
item. If you have an existing keypair, this option will overwrite
it. Once you've generated a keypair, you will need
to [upload the public key to NIM](https://www.nersc.gov/users/connecting-to-nersc/connecting-with-ssh/#toc-anchor-2).
This keypair will be good for 12 hours. You'll need to refresh it if it
expires. You can do this by selecting the "Renew NX SSH Keypair" from
the menu at the lower left hand side.

### Suspending or Terminating a NX Session

When you close the NX window (e.g., by clicking the "cross" button) a
dialog box will appear providing the choice of either suspending or
terminating the session.

*Suspending* the session will preserve most running applications inside
the session intact and allow you to reconnect to the session at a
later time.

*Terminating* the session will kill all the running applications inside
the session and all unsaved work will be lost.

If you lose your connection to the NX server (e.g., if your internet
connection is lost) NX will automatically suspend the session allowing
you to reconnect to the same session while keeping the running
applications intact
