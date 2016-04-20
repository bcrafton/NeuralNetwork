NeuralNetwork
Cuda implementation for Neural network from Stanford's machine learning course on Coursera. 

===Compilation===
This is written for nvcc.

It can be compiled with the following shell command:
nvcc --gpu-architecture=sm_20 --relocatable-device-code=true main.cu matrix.cu vector.cu matrix_list.cu device_matrix.cu device_matrix_list.cu device_vector.cu buffer.cu matrix_util.cu device_matrix_util.cu NeuralNetwork.cu

===Running the Code===
The user DOES NOT need to specify the number of iterations that they would like to do.

Example: To run
./a.out

===Running the code on the discovery cluster===
In this repo there is a script called NN_cuda.bash. In this script I include both the compile 
shell command and the command to execute the code for 100 iterations. The user only needs 
to modify the script for their name and directory on the discovery cluster and execute:
bsub< NN_cuda.bash

This code does not actually compute the neural network ... the implementation was a bit much and was not successful
