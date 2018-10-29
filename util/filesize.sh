#!/bin/bash

for f in $(git diff master --name-only); do
    wc -c $f
done

for f in $(git diff master --name-only); do    
    fs=$(wc -c $f | awk '{print $1}')
    if [ $fs -gt 1048576 ]; then
	echo $f 'is over 1M'
	exit 1
    fi
done
