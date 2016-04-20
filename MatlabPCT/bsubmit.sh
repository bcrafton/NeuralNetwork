#!/bin/bash

# enable your environment, which will use .bashrc configuration in your home directory
#BSUB -L /bin/bash

# the name of your job showing on the queue system
#BSUB -J MatlabJob.01

# the following BSUB line specify the queue that you will use,
#BSUB -q ht-10g

# the system output and error message output, %J will show as your jobID
#BSUB -o %J.out
#BSUB -e %J.err

# the CPU number that you will collect (Attention: each node has 2 CPU)
#BSUB -n 1

# enter the work directory - this should be in scratch
work=/home/crafton.b/NeuralNetwork/MatlabPCT/

# enter the matlab file which is to be placed or copied into your work directory
Matlab_infile=ex4

cd $work
matlab  -logfile ./output.txt -nodisplay -r $Matlab_infile

