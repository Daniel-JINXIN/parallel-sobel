CC=gcc
LD=gcc
CFLAGS = -O2 -std=c99 -Wall -Wextra -c -fopenmp
LDFLAGS = -lgomp

EXES = omp_version cuda_version seq_version

tmp: CFLAGS = -O0 -g -Wall -Wextra -std=c99
tmp: seq_version

all: $(EXE)

dev: all

omp_version: implem_omp.o common.o
	$(LD) $(LDFLAGS) $^ -o $@

seq_version: implem_seq.o common.o
	$(LD) $(LDFLAGS) $^ -o $@

cuda_version: implem_cuda.o common.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o: %.cu


clean:
	rm -f $(EXES)
	rm -f *.o
