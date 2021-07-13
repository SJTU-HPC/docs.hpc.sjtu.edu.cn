#!/bin/bash
grep -nr '\](/' --include="*rst" docs
test $? = 1
