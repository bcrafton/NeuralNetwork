#!/bin/sh

#BSUB -J brian_cuda
#BSUB -o output
#BSUB -e error
#BSUB -n 1
#BSUB -q par-gpu
#BSUB cwd /home/crafton.b/cuda_nn
cd /home/crafton.b/cuda_nn

rm output
rm error

nvcc --gpu-architecture=sm_20 --relocatable-device-code=true main.cu matrix.cu vector.cu matrix_list.cu device_matrix.cu device_matrix_list.cu device_vector.cu buffer.cu matrix_util.cu device_matrix_util.cu neural_network.cu
./a.out