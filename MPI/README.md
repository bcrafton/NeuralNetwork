# Sequential Implementation
This is the sequential implementation of the Neural Network
### Compilation
This is written for mpiCC.

It can be compiled with the following shell command: </br>
**mpiCC -o a.out main.c matrix.c matrix_list.c vector.c neural_network.c matrix_util.c**
### Running the Code
The user DOES NOT NEED to specificy number of iterations that they would like to do</br>
Example: To run gradient descent </br>
**mpirun -np 16 -prot -TCP -lsf ./a.out**
### Running the code on the discovery cluster
In this repo there are 4 scripts called nn_mpi4.bash, nn_mpi8.bash, nn_mpi16.bash, and nn_mpi32.bash. In these script I include both the compile shell command and the command to execute the code for 100 iterations. The user only needs to **modify the script for their name and directory** on the discovery cluster and execute:</br>
**bsub< nn_mpi32.bash**
