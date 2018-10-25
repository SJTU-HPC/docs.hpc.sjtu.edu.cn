# Collaboration Accounts On Genepool

## Overview

The production computing environment on the system has been
set up to allow, upon request, collaboration accounts to be created.
The purpose of these collaboration accounts is to allow collections of
users to equally access and manipulate files and jobs run by the
collaboration user.

## Requesting and Maintaining Collaboration Accounts

PIs, PI proxies, and JGI group leads can request
collaboration accounts.  Please file a ticket to request a
collaboration account by visiting https://help.nersc.gov to file a
ticket.  Furthermore, only PIs, PI proxies and JGI group
leads can request changes in membership to collaboration accounts.
Please file a service ticket to change collaboration account
membership or host settings.

## Using a Collaboration Account

Collaboration accounts on genepool will allow permitted users on
specific hosts to switch users.  To switch users on genepool, first
ssh to your group gpint system and then run the `collabsu` command,
and finally enter your NIM password to gain access to the
collaboration account.  `collabsu` is a replacement for sudo which
allows user-level secured switching on the diverse and complex
genepool environment.

```sh
mamelara@denovo:~$ ssh gpint
…
mamelara@gpint13:~$ collabsu annotrub
[sudo] password for mamelara:
annotrub@gpint13:~$
```

## Restrictions on Collaboration Accounts

Collaboration accounts are a special class of account which do not
allow direct password access.  If you have a legacy collaboration
account which does allow password access, please expect us to contact
you about converting it to a modernized and secure collaboration
account.  If you do have password access to a legacy collaboration
account, please remember that it not permissible for users to share
passwords for NERSC accounts, nor is it permissible to hold multiple
NERSC accounts.

Collaboration accounts are afforded the same privileges as other
NERSC-user accounts, including the same quota limits on /global/homes
and scratch directories. It is important to be careful that the
collaboration account does not exceed quotas.  Maintaining
coordination between multiple users can be challenging, so ensure you
are communicating regularly with your co-users of the collaboration
account.
