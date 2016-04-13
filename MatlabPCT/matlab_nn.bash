#!/bin/bash                                                                                                                                                                               
#BSUB -L /bin/bash                                                                                                                                                                        
#BSUB -J brian                                                                                                                                                                   
#BSUB -q ht-10g                                                                                                                                                                           
#BSUB -o output                                                                                                                                                                           
#BSUB -e error                                                                                                                                                                           
#BSUB -n 30                                                                                                                                                                               
work=/home/crafton.b/matlab_nn
cd $work
matlab -logfile /home/crafton.b/matlab_nn/out.txt -nodisplay -r "ex4()"
