#!/bin/sh
#BSUB -J BrianHello
#BSUB -o output_file
#BSUB -e error file
#BSUB -n 1
#BSUB -q ht-10g
#BSUB cwd .

gcc -g main.c matrix.c matrix_util.c neural_network.c vector.c matrix_list.c -lm
./a.out 100
