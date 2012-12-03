CC=gcc
LD=gcc
NVCC=nvcc
NVCCFLAGS = -c -arch=sm_11
INCLUDES = -I. -I./third_party/lodepng/
#XXX: -std=gnu99 for PAPI only, go back to c99 later !
CFLAGS = -O3 -std=gnu99 -Wall -Wextra -c -fopenmp $(INCLUDES)
LDFLAGS = -lgomp -lm -lcudart

# We won't be fine-grained in dependance to headers and consider
# that everything depends on every header.
HEADERS = $(wildcard *.h)
TESTS_C = $(wildcard tests/*.c)
TESTS = $(patsubst %.c, %, $(TESTS_C))

EXES = omp_version seq_version cuda_version_recursive

all: $(EXES)

tests: build_tests
	@./run_tests.sh

build_tests: $(TESTS)

dev: CFLAGS = -g -O0 -std=gnu99 -Wall -Wextra -c -fopenmp $(INCLUDES)
dev: NVCCFLAGS = -c -arch=sm_11 -g
dev: all

LODEPNG_OBJ = ./third_party/lodepng/lodepng.o

omp_version: implem_omp.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_omp.o common.o $(LODEPNG_OBJ) -o $@

seq_version: implem_seq.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_seq.o common.o $(LODEPNG_OBJ) -o $@

cuda_version: implem_cuda.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_cuda.o common.o $(LODEPNG_OBJ) -o $@

cuda_version_linear: implem_cuda_linear.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_cuda_linear.o common.o $(LODEPNG_OBJ) -o $@

cuda_version_recursive: implem_cuda_recursive.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_cuda_recursive.o common.o $(LODEPNG_OBJ) -o $@

# a test file depends on itself AND the corresponding implem in toplevel dir,
# common.c, and lodepng. But it includes C files, to no need to link it
# with other objects (except lodepng)
tests/%_test: tests/%_test.o $(LODEPNG_OBJ) $(HEADERS) %.c common.c tests/%_test.c
	$(LD) $(LDFLAGS) $< $(LODEPNG_OBJ) -o $@


common.o: $(LODEPNG_OBJ)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(EXES)
	rm -f *.o
	rm -f implem_seq_test
	rm -f out.png
	rm -f tests/*.o
	rm -f $(TESTS)

cleanall: clean
	rm -f third_party/lodepng/*.o

tags:
	ctags -R

.PHONY: clean tags tests cleanall
