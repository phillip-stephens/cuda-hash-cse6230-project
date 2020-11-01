.PHONY: all clean

# CC = clang -Xpreprocessor -fopenmp -lomp
#nvcc  ./mm.cu -Xcompiler -fopenmp
# LIBS  = -lomp
# CFLAGS = -O2
CC = nvcc
LIBS  =
CFLAGS = -O2 -Xcompiler -fopenmp
CXX = g++-10.1.0
CXXLIBS = -lpthread
CXXFLAGS = -std=c++11

#SRC=$(wildcard *.c)
SRC= md5.cu
EXE = $(patsubst %.cu,%.exe,$(SRC))

all: 
	nvcc  ./md5.cu -Xcompiler -fopenmp -lineinfo -o md5.exe
debug:
	nvcc  ./md5.cu -g -G -Xcompiler -fopenmp -lineinfo -o md5.exe


clean:
	rm -f *.exe

%.exe: %.c Makefile
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)