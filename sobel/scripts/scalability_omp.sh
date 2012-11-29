#!/bin/sh

for i in 1 2 `seq 3 3 48`
do
# to average results
    for j in `seq 1 2`
    do
        OMP_NUM_THREADS=$i ./omp_version images/eiffel1000.png /dev/null perfs/scalability_omp_${i}_threads_${j}.log "Scalability for Sobel, OpenMP version"
    done
done
