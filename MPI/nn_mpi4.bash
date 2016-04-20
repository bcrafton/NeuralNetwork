#!/bin/sh
#BSUB -J Brian-Hello
#BSUB -o output4
#BSUB -e error4
#BSUB -n 4
#BSUB -R "span[ptile=16]"
#BSUB -q ht-10g
#BSUB cwd .
cd .

rm error4
rm output4

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
mpirun -np 4 -prot -TCP -lsf ./a.out