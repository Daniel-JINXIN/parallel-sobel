CC=gcc
LD=gcc
INCLUDES = -I. -I./third_party/lodepng/
CFLAGS = -O3 -std=c99 -Wall -Wextra -c -fopenmp $(INCLUDES)
LDFLAGS = -lgomp -lm

# We won't be fine-grained in dependance to headers and consider
# that everything depends on every header.
HEADERS = $(wildcard *.h)
TESTS_C = $(wildcard tests/*.c)
TESTS = $(patsubst %.c, %, $(TESTS_C))

EXES = omp_version cuda_version seq_version

tmp: CFLAGS = -O0 -g -Wall -Wextra -std=c99 -c -fopenmp $(INCLUDES)
tmp: seq_version

all: $(EXE)

tests: build_tests
	@./run_tests.sh

build_tests: $(TESTS)

dev: all

LODEPNG_OBJ = ./third_party/lodepng/lodepng.o

omp_version: implem_omp.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_omp.o common.o $(LODEPNG_OBJ) -o $@

seq_version: implem_seq.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_seq.o common.o $(LODEPNG_OBJ) -o $@

cuda_version: implem_cuda.o common.o $(LODEPNG_OBJ) $(HEADERS)
	$(LD) $(LDFLAGS) implem_cuda.o common.o $(LODEPNG_OBJ) -o $@


# a test file depends on itself AND the corresponding implem in toplevel dir,
# common.c, and lodepng. But it includes C files, to no need to link it
# with other objects (except lodepng)
tests/%_test: tests/%_test.o $(LODEPNG_OBJ) $(HEADERS) %.c common.c tests/%_test.c
	$(LD) $(LDFLAGS) $< $(LODEPNG_OBJ) -o $@


common.o: $(LODEPNG_OBJ)

%.o: %.cu

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(EXES)
	rm -f *.o
	rm -f third_party/lodepng/*.o
	rm -f implem_seq_test
	rm -f out.png
	rm -f tests/*.o
	rm -f $(TESTS)

tags:
	ctags -R

.PHONY: clean tags tests
