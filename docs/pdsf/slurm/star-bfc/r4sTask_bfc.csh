#!/bin/tcsh 
# Note, all code in this scrpt is executed under tcsh
if (  "$SLURM_JOB_PARTITION" =~ *"chos"  ) then  
    echo  task-in-chos
    chosenv
    ls -l /proc/chos/link
else
    echo  task-in-shifter
    echo inShifter:`env|grep  SHIFTER_RUNTIME`
    cat /etc/*release
    #
    # - - - -  D O   N O T   T O U C H  T H I S   S E C T I O N- - - - 
    #
    whoami    
    echo  load STAR enviroment in shifter
    set NCHOS = sl64
    set SCHOS = 64
    set DECHO = 1
    set SCRATCH = $WRK_DIR/out-star1
    setenv GROUP_DIR /common/star/star${SCHOS}/group/
    source $GROUP_DIR/star_cshrc.csh    
    # Georg fix for tcsh
    setenv LD_LIBRARY_PATH /usr/common/usg/software/gcc/4.8.2/lib:/usr/common/usg/software/java/jdk1.7.0_60/lib:/usr/common/usg/software/gcc/4.8.2/lib64:/usr/common/usg/software/mpc/1.0.3/lib/:/usr/common/usg/software/gmp/6.0.0/lib/:/usr/common/usg/software/mpfr/3.1.3/lib/:$LD_LIBRARY_PATH
    echo
    echo avaliable STAR-lib version in this OS image:
    ls -d /common/star/star64/packages/SL*
    #
    # - - - -   Y O U   C A N   C H A N G E   B E L O W  - - - -
    #    
endif  

    #cd ${WRK_DIR} # important- not any more, shifter was fixed
    set daqN = $argv[1]
     
    echo  starting new-r4s PATH_DAQ=$PATH_DAQ, daqN=$daqN, execName=$EXEC_NAME, NUM_EVE=$NUM_EVE, OUT_DIR=$OUT_DIR, workerName=`hostname -f`, startDate=`date`

    echo "use BFC $BFC_String "
    if ( !  -f $daqN ) then
	echo "ERROR: file ${daqN} does not exist, Aborting-33"
	exit
    endif
    echo size of daqN
    ls -l  $daqN 
     
    echo testing STAR setup $STAR_VER in `pwd`
    starver $STAR_VER 
    env |grep STAR

    echo 'my new STAR ver='$STAR'  test root4star '
    root4star -b -q 
    if ( $? != 0) then
	echo STAR environment is corrupted, aborting job
	echo $STAR
	which root4star
	exit
    endif
 
    #echo EEEEE ;   exit

    echo `date`" Fire: $EXEC_NAME for daqN=$daqN  numEve=$NUM_EVE  [wiat]"
    /usr/bin/time -v  $EXEC_NAME -b -q bfc.C\($NUM_EVE,\"$BFC_String\",\"$daqN\"\) >& $LOG_PATH/r4s-${SLURM_JOB_ID}.log
    echo `date`" completed job $daqN  , save results to "$OUT_DIR
    ls -l
    time cp *MuDst* $OUT_DIR
    echo 'copy done '`date`

