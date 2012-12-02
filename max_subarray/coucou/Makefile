CXX ?= g++-4.7
CXXFLAGS = -Wall -Wextra -O2 -g -std=c++11 -c -fopenmp

LD = g++-4.7
LDFLAGS = -lgomp



all: max_subarray

max_subarray: max_subarray.o
	$(LD) $(LDFLAGS) $^ -o $@


%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@


dev: CFLAGS = -Wall -Wextra -00 -g
dev: all

clean:
	rm -f max_subarray
	rm -f *.o

.PHONY: clean
