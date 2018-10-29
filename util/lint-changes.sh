#!/bin/bash

for f in $(git diff master --name-only); do   
    markdownlint $f
done
