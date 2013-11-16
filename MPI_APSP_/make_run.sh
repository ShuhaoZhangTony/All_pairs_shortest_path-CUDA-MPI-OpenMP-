#!/bin/bash
echo "compile File"
rm APSPtest
mpicc -std=c99 -o APSPtest APSPtest.c MatUtil.c

for num_P in 1 2 4 6 8 10 ; do
	if [ -f APSPtest ]		
	then				
		echo "Run start"	
		qsub -pe mpich $num_P mysge.sh
    else
		echo "File not exist."
	fi				
done
