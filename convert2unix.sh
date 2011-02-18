#!/bin/sh

# Simple script to convert the input file to unix format if 
# it is not a directory and not empty
# Erik Sherwood (sherwood@cam) 16OCT06

if [ -d $1 ]; then
    echo "$1 is a directory, skipping ..."
else
    if [ -s $1 ]; then
	dos2unix $1
    fi
fi
