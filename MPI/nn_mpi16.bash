#!/bin/sh
#BSUB -J Brian-Hello
#BSUB -o output16
#BSUB -e error16
#BSUB -n 16
#BSUB -R "span[ptile=16]"
#BSUB -q ser-par-10g
#BSUB cwd .
cd .

rm error16
rm output16

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
mpirun -np 16 -prot -TCP -lsf ./a.out