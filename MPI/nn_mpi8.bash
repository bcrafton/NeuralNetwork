#!/bin/sh
#BSUB -J Brian-Hello
#BSUB -o output8
#BSUB -e error8
#BSUB -n 8
#BSUB -R "span[ptile=16]"
#BSUB -q ht-10g
#BSUB cwd .
cd .

rm error8
rm output8

tempfile1=hostlistrun
tempfile2=hostlist-tcp
echo $LSB_MCPU_HOSTS > $tempfile1
declare -a hosts
read -a hosts < ${tempfile1}
for ((i=0; i<${#hosts[@]}; i += 2)) ;
 do
 HOST=${hosts[$i]}
 CORE=${hosts[(($i+1))]}
 echo $HOST:$CORE >> $tempfile2
done

mpiCC -o a.out main.c matrix.c matrix_list.c vector.c neural_network.c matrix_util.c
mpirun -np 8 -prot -TCP -lsf ./a.out