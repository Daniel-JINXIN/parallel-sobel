#/bin/sh

export OMP_NUM_THREADS=$1

echo -n "sequential version: "
./seq_version images/eiffel1500.png /dev/null
echo -n "parallel version: "
./omp_version images/eiffel1500.png /dev/null
