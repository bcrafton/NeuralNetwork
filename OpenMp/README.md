# OpenMp
This is the OpenMp implementation for the Neural Network 
### Compilation
This is written for compilation with Open Mp support on top of C

It can be compiled with the following shell command: </br>
**g++ -fopenmp -o a main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm**
### Running the Code
The user also needs to specify the number of iterations that they would like to do, this is done by adding an integer after running the binary.</br>
Example: To run gradient descent for 100 iterations add '100' after running the binary </br>
**./a.out 100**
### Running the code on the discovery cluster
In this repo there is a script called NN.bash. In this script I include both the compile shell command and the command to execute the code for 100 iterations. The user only needs to **modify the script for their name and directory** on the discovery cluster and execute:</br>
**bsub< NN_open_mp.bash**

###Results
Will be added when I have time to format them for this markdown language.
