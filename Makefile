use: coo_matrix.o use.o 
	nvcc -arch=sm_20 use.o coo_matrix.o -o use -lcusparse
use.o: use.cu coo_matrix.h
	nvcc -arch=sm_20 -c use.cu
coo_matrix.o: coo_matrix.cu coo_matrix.h
	nvcc -c -arch=sm_20 coo_matrix.cu -lcusparse

clean:
	rm -f *.o use
