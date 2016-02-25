# NeuralNetwork
C implementation for Neural network from Stanford's machine learning course on Coursera. 
### Compilation
This is written for gcc c compiler.

It can be compiled with the following shell command: </br>
**gcc main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm**
### Running the Code
The user also needs to specify the number of iterations that they would like to do, this is done by adding an integer after running the binary.</br>
Example: To run gradient descent for 100 iterations add '100' after running the binary </br>
**./a.out 100**
### Running the code on the discovery cluster
In this repo there is a script called NN.bash. In this script I include both the compile shell command and the command to execute the code for 100 iterations. The user only needs to **modify the script for their name and directory** on the discovery cluster and execute:</br>
**bsub< NN.bash**
###Data
The training data and some already created matrices are available in the form of csv files included in this repo. As is, the program is using the premade matrices to eliminate randomness for debugging and for grading.

**X.csv** contains the training data. Each row is a 20x20 greyscale image. So there are 400 columns (400 indexes in each row) and each one contains a pixel. There are 5000 rows, meaning 5000 training images.

**y.csv** contains what the images actually are, because this is a supervised learning problem it is necessary to have the actual classification of the data that is being trained. y.csv is a 5000x1 matrix. So there are 5000 rows, each of which contains the actual solution for the respective X row. So the 20x20 image in each row of X maps to the actual classification of that image in y.
Example: Row 20 in X contains 400 pixels that represent a handrawn '2'. Row 20 of y will contain the number 2.

**theta1.csv and theta2.csv** contain premade random classifier data. This is for debugging and grading. The user can choose to use the premade matrix or create their own random matrix in runtime with the 'matrix_random' function.
