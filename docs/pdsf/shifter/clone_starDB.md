This instruction shows how to **transfer tables from a live STAR DB into static directory**, to be uses by the  Shifter-based local STAR DB serving  tables  to root4star jobs running on Cori/Edison.

The key instruction doing the trick is:
```bash
 mysqldump -h liveStarDB  --all-databases  |   mysql -h staticDB
```

The main challange is transfer of about 40GB of data thrugh this pipe. The instruction below describes how to execute it in a stable, reproducible fashion.

The commands  I have executed  are shown below. You may use different machines and directory paths - so this instruction should not be taken literary. The outputs of most commands are listed for completeness.



## 0) Prerequisites

0.1) Access to a machine in proximity to the live STAR DB and about 45 GB of free disc space.
I have **ssh pdsf6** , read tables from  **mstardb02**, and wrote output to my **/project** area.
But the same could be accomplished on a rcfnnn machine talking to BNL master DB, writing to some STAR disc at RCF.

0.2) A Shifter image w/ base mysql5.1 software needs to exist at Cori. At the time of writing the image name is **--image=balewski/mysql51-balewski:c**.  This image does NOT conatin any STAR DB payload, so it is unchanged by the process described in this instruction. It is only a reusable tool.

0.3) A current (old) STAR DB data-vault which works with the above shifter image. At the moment it is
``` bash
33G	/project/projectdirs/mpccc/balewski/myStarDB/mysql51Vault-2018-02-22d3d
```


## 1) Dump live STAR DB content as one big file
```bash
ssh pdsf6
pdsf6 $ cd /project/tmpDir
pdsf6 $ time mysqldump -h mstardb02 -P3316 -u loadbalancer -p --all-databases >dump.txt2
pdsf6 $ ls -l
43160699328 Aug  8 07:42 dump.txt2

```

You will be asked for the mysql priviledged password - Jeff or Dmitry will give it to you.
This step takes about 9 minutes of wall time, can be done interactively.

Note, I have executed step 1 on pdsf because the next step 2 executed on Cori will read the dump.txt2 and project is seen on Cori. But you could execute step 1 at RCF, and just scp dump.txt2 to Cori-scratch afterwards.

## 2) Create new STAR DB-vault with updated content.
This steps takes 4 hours total, so use 'screen' as indicated. Remember on which Cori node you start the screen. (If you are not familiar w/ screen consult Google.)

2.1) Clone your working (old) DB-vault and place the copy on Cori-cscratch, pick a meaningfull new name.
```bash
cori12$ scp -rp /project/projectdirs/mpccc/balewski/myStarDB/mysql51Vault-2018-02-22d3d /global/cscratch1/sd/balewski/starDB-master/mysql51Vault-2018-08-08
```
It takes about 2 minutes of wall time.

2.2) Launch Shifter image with mysql5.1 and volumemount the newly clonned DB-vault. Start mysqld deamon. Verify you can talk to your DB by executing simple querry.

Note, currently msqld can survive logout from Cori, so always check if the old mysqld deamon is perhaps still alive and if it is kill it.

```bash
ssh cori
cori12$ lsof -i -P -n
COMMAND   PID     USER   FD   TYPE    DEVICE SIZE/OFF NODE NAME
mysqld  32034 balewski    3u  IPv4 192470244      0t0  TCP *:3306 (LISTEN)
cori12$ kill -9 32034
cori12$ lsof -i -P -n
cori12$
```

**Start new Shifter image under screen**
```bash
cori12$ screen -S janStar
cori12$ shifter  --volume=/global/cscratch1/sd/balewski/starDB-master/mysql51Vault-2018-08-08:/mysqlVault  --image=balewski/mysql51-balewski:c bash

### start new mysqld, wait at least 15 seconds after this command
cori12$ /usr/bin/mysqld_safe  --defaults-file=/mysqlVault/my.cnf &
180809 12:27:55 mysqld_safe mysqld process hanging, pid 41466 - killed
180809 12:27:55 mysqld_safe mysqld restarted
bash-4.1$

### this is good output:
bash-4.1$ ps
  PID TTY          TIME CMD
37226 pts/42   00:00:00 bash
43319 pts/42   00:00:00 bash
44684 pts/42   00:00:00 mysqld_safe
44790 pts/42   00:00:00 mysqld
45533 pts/42   00:00:00 ps
bash-4.1$
```

Verify you can talk to this DB
```bash
bash-4.1$ mysql -u balewski --socket=/mysqlVault/mysql.sock -pjan -e 'SELECT user, host FROM mysql.user;'
+--------------+--------------+
| user         | host         |
+--------------+-------------- +
| %            | %            |
| balewski     | %            |
| loadbalancer | %            |
| root         | 127.0.0.1    |
```

The most time consuming step: pipe the 40GB of data to the running DB, takes about 4 hours.
**By inside screen for this operation.**
```bash
bash-4.1$ time cat dump.txt2 |   mysql -u balewski --socket=/mysqlVault/mysql.sock -pjan
real    264m46.689s
```

You are done. Just clean up.
```bash
### verify DB still works, e.g. list all table
bash-4.1$ mysql -u balewski --socket=/mysqlVault/mysql.sock -pjan -e 'show databases'
+----------------------------+
| Database                   |
+----------------------------+
| information_schema         |
| Calibrations               |
| Calibrations_eemc          |
| Calibrations_emc           |
| Calibrations_epd           |

### shutdown the daemon
bash-4.1$ pkill -9 mysqld
[1]+  Killed                  /usr/bin/mysqld_safe
bash-4.1$ ps
  PID TTY          TIME CMD
 8791 pts/42   00:00:00 ps
37226 pts/42   00:00:00 bash
43319 pts/42   00:00:00 bash

### exit shifter
bash-4.1$ exit
exit
balewski@cori12:~>
```

**Archive your new DB-valut** so it is not flusshed from Cori-scratch after 12 weeks.
``` bash
@cori12:~> du -hs /global/cscratch1/sd/balewski/starDB-master/mysql51Vault-2018-08-08
33G	/global/cscratch1/sd/balewski/starDB-master/mysql51Vault-2018-08-08

cori12:~> cp -rp /global/cscratch1/sd/balewski/starDB-master/mysql51Vault-2018-08-08 /project/projectdirs/mpccc/balewski/myStarDB/

cori12:~> du -hs /project/projectdirs/mpccc/balewski/myStarDB/*
34G	/project/projectdirs/mpccc/balewski/myStarDB/mysql51Vault-2018-08-08
```

Note, the /project directory size may be different - it is OK, not sure why. perhaps Luster is changing block size.













