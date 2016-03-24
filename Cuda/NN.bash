#!/bin/sh
#BSUB -J BrianHello
#BSUB -o output_file
#BSUB -e error file
#BSUB -n l
#BSUB -q ht-10g
#BSUB cwd /home/crafton.b/NN/src/

work=/home/crafton.b/NN/src/
cd $work

gcc -g main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm
./a.out 100
