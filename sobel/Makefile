CC=gcc
LD=gcc
INCLUDES = -I. -I./third_party/lodepng/
CFLAGS = -O2 -std=c99 -Wall -Wextra -c -fopenmp $(INCLUDES)
LDFLAGS = -lgomp

EXES = omp_version cuda_version seq_version

tmp: CFLAGS = -O0 -g -Wall -Wextra -std=c99 -c -fopenmp $(INCLUDES)
tmp: test

all: $(EXE)

dev: all

LODEPNG_OBJ = ./third_party/lodepng/lodepng.o

omp_version: implem_omp.o common.o $(LODEPNG_OBJ)
	$(LD) $(LDFLAGS) $^ -o $@

seq_version: implem_seq.o common.o $(LODEPNG_OBJ)
	$(LD) $(LDFLAGS) $^ -o $@

cuda_version: implem_cuda.o common.o $(LODEPNG_OBJ)
	$(LD) $(LDFLAGS) $^ -o $@

test: common.o $(LODEPNG_OBJ)
	$(LD) $(LDFLAGS) $^ -o $@

common.o: $(LODEPNG_OBJ)

%.o: %.cu

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(EXES)
	rm -f *.o
	rm -f third_party/lodepng/*.o
