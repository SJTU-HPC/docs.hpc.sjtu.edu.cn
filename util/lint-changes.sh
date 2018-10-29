#!/bin/bash

for f in $(git diff master --name-only); do
    if [ ${f: -3} == ".md" ]; then
	markdownlint $f
    fi
done
