#!/bin/sh
#BSUB -J BrianHello
#BSUB -o output_file
#BSUB -e error file
#BSUB -n 32
#BSUB -q ht-10g
#BSUB cwd .

g++ -fopenmp -o a main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm
./a 100 4

./a 100 8

./a 100 16

./a 100 32