# NeuralNetwork

C implementation for Neural network from Stanford's machine learning course on Coursera. 

Written for gcc c compiler.

Training data and actual responses supplied in the csv files

can compile with

gcc main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm

Run this code with specificing the number of iterations to do gradient descent with:

this will run with 100 iterations

./a.out 100
