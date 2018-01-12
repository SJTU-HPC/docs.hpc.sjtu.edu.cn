## Minimal overview of SLURM at PDSF
(see any generic SLURM tutorial for more details)

To submit in SLURM the equivalent of 'qsub  jobscript.sh' is:
```bash
[laptop]
$ ssh  -X pdsf.nersc.gov
sbatch -p shared-chos   jobscript.sh
   Submitted batch job 102992
```

Note 1:  your default chos will be used to run  jobscript.sh , the default run time is set to 24h, default RAM is set to 4 GB<br>

The simplest slurm job  looks like this:
```shell
$cat hello2.slr
{!pdsf/slurm/hello2.slr!}
```
The slumr job is submitted with command
```shell
$sbatch hello2.slr
```
and it will produce the stdout/err as one file: 'slurm-3726343.out', where job id is the big number.

Submit a job array of size 100:
```bash
$ sbatch -p shared-chos    --array=1-100 jobscript.sh
```

Submit a job array of size 100 but run up to 10 tasks at once
```bash
$ sbatch -p shared-chos  -t 24:00:00  --array=1-100%10 jobscript.sh
```

Submit one task running on 32 vCores and use 50 GB of RAM
```bash
$ sbatch -p shared-chos --mem 50000M -n32  jobscript.sh
```

Start **interactive session** on a SLURM worker node with
```bash
$ salloc  -p shared-chos  -t 1:00:00
    salloc: Granted job allocation 93574
```

**Licenses**: optional constraint informing SLURM about resources your job needs. If speciffied will all SLURM to protect ( e.g. not start)  your job in the case given resource is not avaliable. Typically, users who need /project(a) should add this line to the slurm job description
```bash
#SBATCH -L project
#SBATCH -L projecta
```

STAR specific: request license to access HPSS:
```bash
#SBATCH -L starhpssio 
```

Check if you can run jobs in PDSF SLURM
```bash
$ sacctmgr show assoc where user=$USER
     Cluster    Account       User    Share
       pdsf1       lz       balewski    10
```

List all yours queued and running jobs  w/ sqs , no arguments. <br>
 sqs can  also list jobs for other users, see 'sqs --help'. e.g.
```bash
$ sqs -u dybspade
JOBID              ST   REASON       USER         NAME         NODES        USED         REQUESTED    SUBMIT                PARTITION    RANK_P       RANK_BF
20105              R    None         dybspade     rmq_pdsf_kup 1            19:51:39     24:00:00     2017-06-22T13:39:51   shared       N/A          N/A         
20106              R    None         dybspade     rmq_pdsf_kup 1            19:50:02     24:00:00     2017-06-22T13:41:28   shared       N/A          N/A         
```
To learn more  info about one job  you can use Rebecca's line:
```bash
$ sacct --format=job,user,submit,start,end,exitcode,nnodes,alloccpus,timelimit,cputime,state%20,maxvmsize,qos,maxrs -j 21115
       JobID      User              Submit               Start                 End ExitCode   NNodes  AllocCPUS  Timelimit    CPUTime      State  MaxVMSize        QOS 
------------ --------- ------------------- ------------------- ------------------- -------- -------- ---------- ---------- ---------- ---------- ---------- ---------- 
21115          kkrizka 2017-06-23T13:23:53 2017-06-23T13:23:53 2017-06-23T13:32:54      0:0        1          1   00:25:00   00:09:01  COMPLETED                normal 
21115.batch            2017-06-23T13:23:53 2017-06-23T13:23:53 2017-06-23T13:32:54      0:0        1          1              00:09:01  COMPLETED    130940K            
```

Why my job is not starting?
```bash
$ scontrol show job 28547_300
   JobId=28547 ArrayJobId=28547 ArrayTaskId=300 JobName=atlas-chos
   Priority=1802 Nice=0 Account=atlas QOS=normal
   JobState=PENDING Reason=Resources Dependency=(null)
   Partition=shared-chos AllocNode:Sid=pdsf8:15532
   NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=1,mem=3008,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=2 MinMemoryNode=3008M MinTmpDiskNode=0
```
How many any-jobs are running in shared-chos partition?
```bash
$ squeue -p shared-chos |nl|tail
     1	 JOBID     USER ACCOUNT           NAME  ST REASON          START_TIME                TIME  TIME_LEFT NODES CPUS  PARTITION   PRIORITY
     3	34047_  shapiro   atlas     atlas-chos   R None            2017-06-29T10:51:53    1:10:42    3:44:18     1    1 shared-cho        722
     4	34047_  shapiro   atlas     atlas-chos   R None            2017-06-29T10:51:53    1:10:42    3:44:18     1    1 shared-cho        722
```
How many job slots,status,etc are in the queue:
```bash
$ scontrol show partition shared-chos
   PartitionName=shared-chos
   Nodes=mc15[28-34]
   State=UP TotalCPUs=420 TotalNodes=7 SelectTypeParameters=NONE
   DefMemPerCPU=1000 MaxMemPerCPU=2000
```
Who can run jobs on account=star ?
```bash
$ sacctmgr list assoc account=star
   Cluster    Account       User  Partition     Share
---------- ---------- ---------- ---------- --------- 
     pdsf1       star                             736 
     pdsf1       star     aarose                   10 
     pdsf1       star       abha                   10 
     pdsf1       star   afleming                   10 
     pdsf1       star   agafover                   10 
```

List all SLURM jobs from all PDSF users (former sgeusers)
```bash
$ slusers

Current SLURM usage summed over all PDSF users 
   Rjob     Rcpu   Rcpu*h    PDjob    PDcpu      user:account:partition
      5       15     17.3        0        0      balewski nstaff shared
     10       10     15.2        0        0      balewski nstaff shared-cho
     47       47    103.6       20       20      kkrizka atlas shared
      2        2      0.1        0        0      shapiro atlas shared

   Rjob     Rcpu   Rcpu*h    PDjob    PDcpu      account:partition
     49       49    103.7       20       27      atlas shared
      5       15     17.3        0        7      nstaff shared
     10       10     15.2        0        7      nstaff shared-cho

     64       74    136.3       20       27        TOTAL
```


Primary PDSF shares per experiment
```bash
$ sshare -A alice,rhstar,dayabay,majorana,atlas,lz,lux,cuore,pdtheory -l
pdsf7 $ sshare -A alice,rhstar,dayabay,majorana,atlas,lz,lux,cuore,pdtheory
             Account       User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare
-------------------- ---------- ---------- ----------- ----------- ------------- ---------- 
alice                                  495    0.186160   567432591      0.311335
atlas                                  427    0.160587    12506197      0.006862
cuore                                    2    0.000752      377071      0.000207
dayabay                                265    0.099662   193828083      0.106348
lux                                     26    0.009778    24359728      0.013366
lz                                     400    0.150432   383734087      0.210545
majorana                                51    0.019180     8572583      0.004704
pdtheory                                 2    0.000752           0      0.000000
rhstar                                 736    0.276796   611072852      0.335279
```

Examples of intaractive and SLURM batch jobs for all PDSF experiments, updated June, 2017.


### How can I get code for the examples ?
```bash
ssh pdsf
git clone https://bitbucket.org/balewski/tutorNersc
cd tutorNersc/2017-05-pdsf3.0
ls
```

!!!warning
	if your default shell is tcsh - this tutorial will not work on Cori.


### Table 1 

List of all SLURM+Shifter exampels  provide by PDSF users.

----------
|  Experiment	| shifter image 	| Example code 	| author 	| on Cori |  slurm+CHOS \[remarks]|
|----------	|:-------------:	|-------------	|----------:	| ---|---|
| LZ 	|   custom:pdsf-chos-sl64:v4	| /lz-afan \[lz2]  	| Alden Fan 	| no CVMFS|  yes|
 | Majorana|  custom:pdsf-chos-sl64:v4 	| /majorana-mbuuck/ [mj1] 	|  Micah Buuck 	| yes | yes|
 | Majorana|  custom:pdsf-chos-sl64:v4 	| /majorana-dave/  	|   David Tedeschi	| yes | no|
|ATLAS |    custom:pdsf-chos-sl64:v4 	| /atlas-shapiro [at1] | Haichen Wang | no CVMFS| yes|
|  |  | /atlas-kkrizka [at2] | Karol Krizka | yes | yes , [IOn10] |
 |  |  | /atlas-spgriso [at3] | Simone Griso |  no CVMFS | yes, [IOn10] |
| STAR |   custom:pdsf-sl64-star:v6 | /star-balewski [st1] | J.B. | - | yes |
| | |  root4star BFC [st2] | J.B. | yes |yes , [IOy60]|
| DayaBay | docker:balewski/sl64-dayabay:c| /dayabay-balewski [dyb1] |J.B. | YES (only!)|N/A
| DayaBay | custom:pdsf-chos-sl53:v1 | /dayabay-hack [dyb2]  |Robert  Hackenburg | no |yes |
| LUX |  custom:pdsf-sl64-star:v6 | /lux-epease  | Evan Pease | - | yes |

\[lz2]  LZ reconstruction package, uses cvmfs, reads data from /project. See /lz-afan/Readme.

\[mj1] Majorana  data analysis,raw waveforms classifcation, reads data from /project.

\[at1] rootTask1.sh: copies a tar file of a compiled RootCore package, untar it in a running directory, and do rcSetup and run the executable of the package, needs CVMFS, job-array aware, works on /project

\[at2] launch.sh   oneMG.slr: run MadGraph in local scratch, I/O only to /project, job-array aware. 

\[at3] athena_sim1.sh : athena job that runs simulation, uses AtlasProduction releases, cvmfs, and a flag for large memory. Can be run as job array, each tasks sees different subset of events.

\[st1] STAR interactive detailed tutorial, URL: [ws2-interactive-starExp.md](https://bitbucket.org/balewski/tutornersc/src/master/2017-05-pdsf3.0/star-balewski/ws2-interactive-starExp.md)

\[st2]  r4sTask_bfc.csh : makes a sandbox, set starver, runs BFC on a daq file from /project, use DB:mstardbNN.nersc.gov, job-array aware, writes to $SLURM_TMP, saves to /project

\[dyb1] private Docker image, sl64, CERN libs and DYB software compiled inside the image, works only for user=balewski

\[dyb2] generic sl53, intractive  & batch. Does not work on Cori because DYB bins from /common/dayabay/releases/ are needed.

\[IOy60] can run 60 processes on single node even if the system is empty

\[IOn10] can NOT run even 10  processes on a single node  if the system is empty due to IO contention
