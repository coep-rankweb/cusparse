#ifndef __CUDA
#define __CUDA
#include <cstdio>
#include <vector>
#include <cerrno>
#include <algorithm>

#include <cuda.h>
#include <cusparse.h>

#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/copy.h>

#define ABS(x)	(((x) < 0) ? -(x) : (x))	// MOD gave the impression that we were using (or making use of) moduli.

using namespace std;

void init(void);

template <typename T>
struct convergence {
	T bound;

	convergence(T _bound) : bound(_bound) { }

	__host__ __device__ T operator()(const T& x)const{
		return ABS(x) > bound ? 1 : 0;
	}
};

class coo_matrix;
class Vector {
	double *data;
	int size;
public:
	Vector() {}		// Initializing an empty vector. Mostly returned during errors.
	Vector(double val, int size);
	Vector(double **arr, int size);
	Vector(vector<double> &arr);
	Vector(const Vector &v);
	~Vector() {
		cudaFree(data);
		size = 0;
	}

	Vector operator * (double val);
	Vector operator + (Vector &second);
	Vector operator - (Vector &second);

	void createSparse(vector<int> &, int, int defval = 1);

	/* Hack : Bad Implementation. I am reluctant to write iterator wrapper */
	bool check_convergence(double bound);
	void elementwiseInvert(void);		// Again Bad
	Vector getDanglingVector(void);		// Again Bad


	int getSize() { return size; }
	double *getData() { return data; }
	friend ostream &operator << (ostream &, Vector &);

	
	// You will not understand the purpose unless you read the appl
	friend coo_matrix getMatrix(vector<int> &, int, int);
};

class coo_matrix {
	int num_rows, num_cols, nnz;
	double *cooVal;
	int *cooRow;
	int *cooCol;
	int *csrRowPtr;
	cusparseMatDescr_t descr;

public:
	coo_matrix() {
		cooVal = NULL;
		cooRow = cooCol = csrRowPtr = NULL;
	}
	coo_matrix(int nRows, int nCols, int nNZ, double **val, int **row, int **col, int **csr, cusparseMatDescr_t descr) {
		num_rows = nRows; num_cols = nCols; nnz = nNZ;
		cooVal = *val; cooRow = *row; cooCol = *col;
		csrRowPtr = *csr;
		this->descr = descr;
	}
	coo_matrix(vector<int> &, vector<int> &, vector<double> &, int, int);
	coo_matrix(const coo_matrix &temp);
	~coo_matrix() {
		cudaFree(cooVal);
		cudaFree(cooRow);
		cudaFree(cooCol);
		cudaFree(csrRowPtr);
	}

	int getNumRows() { return num_rows; }
	int getNumCols() { return num_cols; }
	int getNNZ() { return nnz; }
	void diagonalize(Vector &arr);
	coo_matrix &operator * (double val);
	coo_matrix operator * (coo_matrix &mat);
	Vector operator * (Vector &arr);
	coo_matrix operator ~ (void);		// Transpose
	friend ostream &operator << (ostream &, coo_matrix &);

	// You will not understand the purpose unless you read the appl
	friend Vector getIntersection(coo_matrix &);
	friend Vector getInduced(Vector &);
};
#endif
