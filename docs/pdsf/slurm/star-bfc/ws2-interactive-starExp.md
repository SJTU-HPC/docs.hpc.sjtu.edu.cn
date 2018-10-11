## PDSF-3  Interactive  STAR  session  using Shifter image w/ any StarLibs

### Worksheet-2

### Prerequisites

* NERSC credentials
* authorized on PDSF login nodes
* added to 'star' repo at PDSF

### 1) Login to PDSF

Login to PDSF as usually and load SLURM module. You must start this
instruction in CHOS=sl64.

```bash
[laptop]
$ ssh  -X pdsf.nersc.gov
chosenv
>>   sl64  # if you do not see 'sl64' fix it.

module load slurm

# check if you can run jobs in PDSF SLURM
sacctmgr show assoc where user=$USER
>>>   Cluster    Account       User    Share
>>>    pdsf1       star       usgstar    10
```

You should see the name of 'account' you belong to and # of shares you
can use at PDSF-3.

### 2) Launch interactive SLURM session on 1 core on PDSF-3

Start your interactive session using the same Shifter image as on
Cori, excute shifter command, verify you are inside Shifter instance

```bash
salloc -n 1 -p shared -t 50:00 --image=custom:pdsf-chos-sl64:v4 --volume=/global/project:/project --account star
>>>salloc: Granted job allocation 2073

$ shifter /bin/tcsh  # change cgroup, VERY IMPORTANT command

$  env|grep  SHIFTER_RUNTIME
>>>SHIFTER_RUNTIME=1
# do you see '=1' ? - good.
```

### 3) Load STAR environment , e.g. SL17d

```bash
[inside shifter instance]
# STAR software was compiled with gcc 4.8, execute:
$ module load gcc/4.8.2

# load STAR enviroment  -  copy/paste this block:
set NCHOS = sl64
set SCHOS = 64
set DECHO = 1
set SCRATCH = ~/
setenv GROUP_DIR /common/star/star${SCHOS}/group/
source $GROUP_DIR/star_cshrc.csh
# Georg hack to fix tcsh:
setenv LD_LIBRARY_PATH /usr/common/usg/software/gcc/4.8.2/lib:/usr/common/usg/software/java/jdk1.7.0_60/lib:/usr/common/usg/software/gcc/4.8.2/lib64:/usr/common/usg/software/mpc/1.0.3/lib/:/usr/common/usg/software/gmp/6.0.0/lib/:/usr/common/usg/software/mpfr/3.1.3/lib/:$LD_LIBRARY_PATH

starver SL17d
```

Verify STAR software is avaliable for this version:

```bash
$ echo $STAR
  /common/star/star64/packages/SL15e

$ root4star -b -q
  *******************************************
  *                                         *
  *        W E L C O M E  to  R O O T       *
  *                                         *
  *   Version   5.34/30     23 April 2015   *
  *                                         *
  *  You are welcome to visit our Web site  *
  *          https://root.cern.ch            *
  *                                         *
  *******************************************

ROOT 5.34/30 (v5-34-30@v5-34-30, Apr 23 2015, 18:31:46 on linux)

CINT/ROOT C/C++ Interpreter version 5.18.00, July 2, 2010
Type ? for help. Commands must be C++ statements.
Enclose multiple statements between { }.
*** Float Point Exception is OFF ***
 *** Start at Date : Mon May  8 11:18:12 2017
QAInfo:You are using STAR_LEVEL : SL17d, ROOT_LEVEL : 5.34.30 and node : mc1504
root4star [0]

This is the end of ROOT -- Goodbye

```

Run short starsim simulation, following STAR drupal instruction
https://drupal.star.bnl.gov/STAR/comp/simu/event-gen

```bash
$ mkdir aaa ; cd aaa
$ ln -s $STAR/StRoot/StarGenerator/macros/starsim.pythia8.C starsim.C
$ head -n5 starsim.C
$ root4star -q -b starsim.C\(10\)

# display tracks Pz
$ root4star -l pythia8.starsim.root
root [1] genevents->StartViewer()
root [2] genevents->Draw("mPz","mStatus>0")
# do you see a plot? Good.
```
Verify you see all directories you may need:
```bash
$ cd ; pwd
$ ls ~/
$ ls  /project/projectdirs/star
$ ls /global/projecta/projectdirs/starprod/daq/2015/pp200_mtd/
```

Compile your private code w/ cons in SL16d

```bash
$ starver SL16d
$ echo $STAR
  /common/star/star64/packages/SL16d
$ mkdir -p aaa/StRoot ; cd aaa
$ cp -rp $STAR/StRoot/St_TLA_Maker StRoot
$ cons
 . . .
  Install .sl64_gcc482/obj/StRoot/St_TLA_Maker/St_TLA_Maker.so as .sl64_gcc482/lib/libSt_TLA_Maker.so
```

Debugger works

```bash
$ gdb root4star
(gdb) r
Starting program: /common/star/star64/packages/SL17d/.sl64_gcc482/bin/root4star
...
ctrl-C
(gdb) where
#0  0x00130430 in __kernel_vsyscall ()
#1  0x035c9910 in raise () from /lib/libpthread.so.0
...
```

### 3) Run STAR BFC on few p-Au events

* check you are in SL64 inside STAR shifter image
* find the .daq files location
* fire root4star
* inspect produced muDst

```bash
[in Shifter image, after STAR software loaded]

$ starver SL16d
$ echo $STAR
   /common/star/star64/packages/SL16d

$ ls -l /global/projecta/projectdirs/starprod/daq/2015/pau200_bht1_adc/st_physics_adc_16150045_raw_5500055.daq

$ mkdir bbb; cd bbb

# fire BFC on 5 events, takes ~3 minutes
$ time root4star -b -q bfc.C'(1,5,"DbV20160418 P2014a pxlHit istHit btof mtd mtdCalib BEmcChkStat -evout CorrX OSpaceZ2 OGridLeak3D -hitfilt", "/global/projecta/projectdirs/starprod/daq/2015/pau200_bht1_adc/st_physics_adc_16150045_raw_5500055.daq")' > & Log1 &

$ top ibn1
  PID USER      PR  NI  VIRT  RES  SHR S %CPU %MEM    TIME+  COMMAND
20836 balewski  20   0  345m 144m  66m R 91.2  0.1   0:04.80 root4star
Wait 3 minutes...

    Done after: 193.668u 1.176s 3:32.15 91.8%	0+0k 54324+0io 164pf+0w
    BFC connected to  :
    St_db_Maker:INFO  - MysqlDb::Connect(host,user,pw,database,port) line=525  Server Connecting: DB=StarDb  Host=mstardb02.nersc.gov:3316

$ ls -lrt  #verify the *.root output file are there

# browse muDst
$ root4star -l st_physics_adc_16150045_raw_5500055.MuDst.root
root4star [0]  MuDst->StartViewer()
root4star [1]  MuDst->Draw("mNHits")

```

### 4) Shut down the interactive session

You need to exit 2x: from shifter cgroup, next from interactive SLURM (aka salloc)

```bash
[mc1504] ~/> exit
$ exit
[bash-4.1$]
$ exit
salloc: Relinquishing job allocation 1989
pdsf6 $
```
