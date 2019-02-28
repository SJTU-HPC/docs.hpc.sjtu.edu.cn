#!/bin/bash
grep -nr '\](/' --include="*md" docs
test $? = 1
