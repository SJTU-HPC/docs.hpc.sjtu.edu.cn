# SSH

All NERSC computers (except HPSS) are reached using either the Secure
Shell (SSH) communication and encryption protocol (version 2) or by
Grid tools that use trusted certificates.

SSH (Secure Shell) is an encryted network protocol used to log into
computers over an unsecured network. On UNIX/LINUX/BSD type sytems,
SSH is also the name of a suite of software applications for
connecting via the SSH protocol. The SSH applications can execute
commands on a remote machine and transfer files from one machine to
another.  All communications are automatically and transparently
encrypted, including passwords. Most versions of SSH provide login
(ssh, slogin) a remote copy operation (scp), and many also provide a
secure ftp client (sftp). Additionally, SSH allows secure X Window
connections.

!!! example
    ```bash
    $ ssh elvis@edison.nersc.gov
    elvis@edison.nersc.gov's password: [enter NIM password for user elvis]
	edison$
    ```

## Passwordless logins and transfers

!!! warning
	All public keys must be stored in [NIM](https://nim.nersc.gov).

## Key fingerprints

 *  Cori
	```
	2048 SHA256:mR3sHwHorgjqRlUbggtfOCa768/uKdbNb2TOH8xDMn8
	```

 *  Edison
	```
	4096 SHA256:cbyxNBzfC7hvN56EpYFnYN/YpLY/cQEverdcbpIYjL8
	```

 *  PDSF
	```
	1024 3d:28:24:53:66:de:30:9e:eb:25:3b:03:b0:24:1c:77
	```

## Host Keys

These are the entries in `~/.ssh/known_hosts`.

### Cori

```
cori.nersc.gov ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCvoau+F7fGIHuvcDZZSG7dD2J7hgo3RupUL6Jaw978mb
P3h2Dt/b8F5EHniGOw1sxYrm3wjerF3I1jTYLM6ORndbw+4FeyVgXiAUTTdKl9suhfDTm2aFry
WanMnbknktNYbzLdyg1SdUMHhlcxXCniuPO7j0JMECkXZvuRBWDeeO8FQWcGrOIorCoU0liWgX
c0NoEs9IzyK2N4ywExwljpMs7vKwasz8qyjHB2aYaj6cHjV2ShCp+aevPdp1jfBtIgJUMkjMEa
+0K4zWM0aDzZEaj7vIlKpUCDAdQf/DsPoj808KOKLw0+Bs0qamX+D7+aXsPVG/jfBY5wSCgjlhqn
```

### Edison

```
edison.nersc.gov ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDpzjkAkaxZS7dCRQeGCDxcdJd
ZykF4mNxjFUKOGcAC9aqv2+5S6gHvjH8PksDUI2G1g9Tln3O1Y5l/bMIoNpDPO7neZe0IXsQKO/HNsBP
kaHmaeZmvuZmnx6JC1SXh/e/YPQ5Kbef3rL0QM0WmlFoBTng3FA+8J9E0uAJjvOjiOxOA+Nsb9niqAwe
vgGDyaosgGD7+y6RyMt38nkNcX3RhYXqgtkzFLqkPYsITz3wLrRAttMBPx7qdlQ/lxEnINj/g+XUpEsv
JyUl8V5ldz8o0ts2MQkb2tuBgBTeL3MDHlxD4Kie40byTmOVYSNlOiWih0rNQPOZmsjr9UqCB+GE/oWF
R/3/gDoGanY42U7I0echn8lTNk6Una40FipL5CElGKjcBOS9PMp2NkGXy5So0xRDrWYP2TRo2ED5r/8v
PtbbJxh/jvN34GWgj3qGLE6HKLcgj8gi2tHx5pBgoo6bLqEbgDlwz1E2ObVuOnSRuXvfdvwUTJ0SZVyt
8gHMETaKpj4Ah3ylEBGtF++x+7W9N3QF37zX2kkFoaOGQBhLvOKhyNoO+Ak0rNmuTZQAa8QBB9p8VpaY
FwEpn+dU37iroNWtXXkaqCC5ke9kkB89U1S0L9AEIZTvwlkndcldMq6MN3Q+PrVBqQDO9Tmmm1384f7w
SHkp5b1LTxtqICFe7FQ==
```

## Troubleshooting

### "Access Denied" or "Permission Denied"

This is likely a username or password problem.

1. Make sure you are using the proper NERSC user name.
1. Log into [NIM](https://nim.nersc.gov) to clear login failures.

!!! note
	If you are still unable to login, contact the Account Support
	Office at 1-800-66-NERSC, menu option 2.

### Host authenticity

This message may appear when a connection to a new machine is first
established:

```
The authenticity of host 'edison.nersc.gov' can't be established.
RSA key fingerprint is <omitted>
Are you sure you want to continue connecting (yes/no)?
```

1. Check that the fingerprints match
   the [list above](#Key-fingerprints).
1. If they match accept
1. If they do not match [let us know](https://help.nersc.gov).

### Host indentification changed

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
...
```

Ensure that your `~/.ssh/known_hosts` file contains the correct entries for
Edison and Cori.

1. open `~/.ssh/known_hosts`
1. remove any lines refering Cori or Edison
1. add the following two lines (or retry connecting and verify that
   you have the correct "fingerprint" from the above list.
