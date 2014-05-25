use: coo_mat.o use.o 
	nvcc -arch=sm_20 use.o coo_mat.o -o use -lcusparse
use.o: use.cu coo_mat.h
	nvcc -arch=sm_20 -c use.cu
coo_mat.o: coo_mat.cu coo_mat.h
	nvcc -c -arch=sm_20 coo_mat.cu -lcusparse

clean:
	rm -f *.o use
