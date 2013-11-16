#!/bin/bash
make clean
make
rm result.txt
        for size in 32 128 256 512 1024
        do
        for numP in 2 4 8 16 32
        do
                ./APSP $size $numP 1 6 0 >> result.txt
        done
        done


