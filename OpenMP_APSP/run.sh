 #!/bin/bash
   echo "compile File"
   rm APSPtest
   rm result* 
   gcc -std=c99 -o APSPtest APSPtest.c MatUtil.c -lm -fopenmp
   for Size in 1200 1800 2400 3000 3600 4200 4800; do
	for numP in 1 2 3 4 5 6 7 8 9 10 11 12; do	
           echo "Size is $Size"
 	   echo "numP is $numP"
           ./APSPtest $Size $numP >> result.txt
	done
   done

