#! /bin/bash
cd ~/autobuild/docs.sjtu.edu.cn
git reset --hard origin/master
git clean -f
git pull
git checkout master
make update