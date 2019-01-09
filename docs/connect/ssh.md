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
	4096 SHA256:35yiNfemgwzHCHFrPGWrJBCCqERqLtOVSrR36s1DaPc
	```

 *  Edison
	```
	4096 SHA256:riR+3TGNnPs0uqJxJBbvPU+JR3e/Z0xUzBRsip3ZOJ8
	```

 *  PDSF
	```
	1024 3d:28:24:53:66:de:30:9e:eb:25:3b:03:b0:24:1c:77
	```

## Host Keys

These are the entries in `~/.ssh/known_hosts`.

### Cori

```
cori.nersc.gov ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCsxGw19ZL8EN+NZ9HhD+O/nATuvgZxcEuy/yXnGqz5wMzJj6rK7TsrdU8rdJNrhZDe3yjpCiKvqkbSKp22jK2/iMAeWDQvYpMgC6KyiNd0hztowtMFJEwb8gVmtkVioqIaf9ufJnOO0LX5A5J/4fQhICfbyPiX8SsjX0p655/kIm3T6hr7t89b4IkRu19/uWufbNaV/mZSFWl7asLKXJNTMhzEn6bsTcAqlm55Tp4NvCe1hvv6OY/vU5luDz09UDmnDfr/uukmVm5aIjtlZBGqbOe7huNJGIWhoGCN/SoArRu9T9c9fjOlRMOHcf0QYMQmxFQnR0TkJZQoJ5N+EYNUIB9dvnJs2mlN0ZEuUU0RwAUOge7RwujiZ2AWp/dV/PNvLGmDVUxiyXC0Uuw57Ga2e49hYisYU/J/NPp9AbHqO8M6kZqYdqWKYueIsM3FDti3vUbjV4J6sL6mOBbxuJpUhUEX5UXxGbR39hDVx9Lsj4dszu+mcBFnDNcpRCDjw3z+hDqdNNpzhIRlbHQErLBWL3vnn2MLnb/3z163gyRtu1iTuR5myBIs9jLDAsX94VbBzKWdCFe22x4Eo6HwB6u+UHlXov0fnBXtAmgwRegc1gQwxi2FXB/ty0q1EO+PYo3fjUVRRb4uqBBIvpFarwtL0T6iYAYgHY11vH9Z2BFAHQ==
```

### Edison

```
edison.nersc.gov ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDQPHTSKic9gj6kyhfSkBWZdMgnESIQVI0vg7eNIkshb+pXGGGxWQXrIX3IL1W9bvwrjGNc2JAd9c5Y1CfV0B6sKHDm70pPTnZ3bkvWwq81bloepzJHWE9NpNd7xIlVaM7sh7fMURD5/e7E46qldrpBBtudJG9ZgmjxdmDSlWBTp6scDqehBZ+yaRg0N7zoEA86synBi/0DEDzqarHlvXwXS5mWBGnlC3ZI5Uz/QHD4K26y86SeKYf2EKUI0am+gcPRbUHjDLyThK+qFGveRe9G41eEw40hxmO/yooYgCjCEKVbfU4Po2uR6qb/p/cpeMLOyZ3itrZce6WMgQxw+8g9MPKWuCKH7SJnJ/70YQyLuSlc98mr3AE6fjcZZy8Uf4ckH49qXTH4ILYPEzqLzI86eM+tJltayUWV9aQVAG6lBn14DyCvAyAfts+RCE8JkGJzcUSu7UILiIrqEyOpMOrZ/z2wM7mYJlAVBbrjT0LO6hXh/ET/npo7mMhotjtptXk9qg7DLUfL647OZvWjxQxZlE6jtpHilOaCcXpY3pXUZTtza7kv3pRbnPmzWU0iKKLmqsjtAT773SIvJ/78MwwqIF4pEBiPx7Ixmf+rHwpQV/P6ADBadpTfx28297ZjzvQZ+gTscBWULxeUFNfZtm+jmpsMGPNJTXAAlyVW13zO9w==
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
