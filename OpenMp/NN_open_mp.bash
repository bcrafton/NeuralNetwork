#!/bin/sh
#BSUB -J NNOpenMP
#BSUB -o output_file
#BSUB -e error file
#BSUB -n 10
#BSUB -R "span[ptile=8]"
#BSUB -q ht-10g
#BSUB cwd .

g++ -fopenmp -o a main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm
./a 100
