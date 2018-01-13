#!/bin/bash 
#SBATCH -N 1   --account=m2811  
#-SBATCH -C haswell --image=custom:pdsf-chos-sl64:v2 
#SBATCH --image=custom:pdsf-chos-sl64:v4  # Edison
#SBATCH  --partition debug -t 00:10:00
#SBATCH -J oneNuwa

echo  this example uses external DB and writes output in the starting dir
export DBCONF_PATH=/global/homes/b/balewski/.my.cnf-v0

shifter  --volume=/global/project:/project  --volume=/global/projecta:/projecta ./run1DayabayShifter.sh 

pdsf6 $ cat run1DayabayShifter.sh 
#!/bin/bash 
echo inShifter:`env|grep  SHIFTER_RUNTIME`
cat /etc/*release
export CHOS=sl64
source ~/.bashrc.ext

echo setting up DYB environment
pushd ~mkramer/projscratch/NuWa-sl64/NuWa-trunk
source setup.sh
pushd gaudi/GaudiRelease/cmt; source setup.sh; popd
pushd dybgaudi/DybRelease/cmt; source setup.sh; popd
popd

pwd
ls -l

#testing input
ls -lh /project/projectdirs/dayabay/data/exp/dayabay/2015/daq/Neutrino/0815/daq.Neutrino.0054964.Physics.EH1-Merged.SFO-1._0188.data

#testing software
root -b -q 

echo use DB config from
ls -l ${DBCONF_PATH}

kupshare=/project/projectdirs/dayabay/releases/NuWa/3.13.0-opt/NuWa-3.13.0/dybgaudi/Production/P14B/share

filepath=/project/projectdirs/dayabay/data/exp/dayabay/2015/daq/Neutrino/0815/daq.Neutrino.0054964.Physics.EH1-Merged.SFO-1._0188.data

outpath=test1.root 
rm -rf ${outpath}

seqfile=stats.5566

time nuwa.py -n 5000 --random=off --repack-rpc=1 --output-stats "file1:${seqfile}" @${kupshare}/runReco @${kupshare}/runTags @${kupshare}/runODM @${kupshare}/runFilters --dbirollback "* =  2017-04-04T00:00:00" -o ${outpath} ${filepath}

echo done-jan-11
# QA of the output
root -b -q qaOneTask.C'("test1.root")'
echo done-jan-22
ls -l
#echo terminating DB
#mv ${DBCONF_PATH} ${DBCONF_PATH}.done
#echo done-jan-33
