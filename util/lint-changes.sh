#!/bin/bash

any_fails=false

for f in $(git diff master --name-only); do
    if [ ${f: -3} == ".md" ]; then
	markdownlint $f
	if [ "$?" -ne 0 ]; then
	    any_fails=true
	fi
    fi
done

if [ "$any_fails" == true ]; then
    exit 1
fi
