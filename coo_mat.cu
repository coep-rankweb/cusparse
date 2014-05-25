#include "coo_mat.h"

cusparseHandle_t handle;

__global__
void add(double *first, double *second, double *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = first[_id] + second[_id];
}

__global__
void sub(double *first, double *second, double *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = first[_id] - second[_id];
}

__global__
void scalar_mult(double *first, double *res, double val, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = first[_id] * val;
}

__global__
void sequence(int *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = _id;
}

__global__
void invert(double *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = (res[_id] ? 1 / res[_id] : 0);
}

// This kernel is for setting non-zero default value
__global__
void setter(double *res, int size, int defval) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = defval;
}

__global__
void dangling(double *first, double *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = first[_id] ? 0 : 1;
}

__global__
void add_columns(double *first, double *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		atomicAdd(&(res[first[_id]]), 1);
}

// Only selects those URLs that contain all words from the query
__global__
void check_intersection(double *first, int num_urls, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = ((res[_id] == num_urls) ? 1 : 0);
}

// Converts an arbitrary vector into a binary vector. Flattens positive integers to 1.
__global__
void make_binary(double *res, int size) {
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(_id < size)
		res[_id] = ((res[_id] == 0) ? 0 : 1);
}

void init() {
	cusparseCreate(&handle);
}

Vector::Vector(double val, int size) {
	cudaMalloc((void **) &data, size * sizeof(double));
	int num_blocks = size % 1024 ? (size / 1024) + 1 : size  / 1024;
	setter<<<num_blocks, 1024>>>(data, size);
	this->size = size;
}

Vector::Vector(double **arr, int size) {
	data = *arr;
	this->size = size;
}

/* As per the specification
   http://stackoverflow.com/questions/2923272/how-to-convert-vector-to-array-c
   -> the elements in the array are stored in contiguous mem locs. */
Vector::Vector(vector<double> &arr) {
	cudaMalloc((void **) &data, arr.size() * sizeof(double));
	cudaMemcpy(data, &arr[0], arr.size() * sizeof(double), cudaMemcpyHostToDevice);
	this->size = arr.size();
}

Vector::Vector(const Vector &v) {
	cudaMalloc((void **) &(this->data), v.size * sizeof(double));
	cudaMemcpy(this->data, v.data, v.size * sizeof(double), cudaMemcpyDeviceToDevice);
	this->size = v.size;
}

Vector Vector::operator * (double val) {
	double *temp;

	cudaMalloc((void **) &(temp), size * sizeof(double));
	cudaMemset(temp, 0, size * sizeof(double));

	int num_blocks = size % 1024 ? (size / 1024) + 1 : size  / 1024;
	scalar_mult<<<num_blocks, 1024>>>(data, temp, val, size);
	//thrust::transform(thrust::cuda::par, data, data + size, data, scaling(val));

	return Vector(&temp, size);
}

Vector Vector::operator + (Vector &second) {
	if(this->size == second.size) {
		double *temp;

		cudaMalloc((void **) &(temp), second.size * sizeof(double));
		cudaMemset(temp, 0, second.size * sizeof(double));

		int num_blocks = second.size % 1024 ? (second.size / 1024) + 1 : second.size  / 1024;
		add<<<num_blocks, 1024>>>(data, second.data, temp, size);

		return Vector(&temp, second.size);
	}
	return Vector();
}

Vector Vector::operator - (Vector &second) {
	if(this->size == second.size) {
		double *temp;

		cudaMalloc((void **) &(temp), second.size * sizeof(double));
		cudaMemset(temp, 0, second.size * sizeof(double));

		int num_blocks = second.size % 1024 ? (second.size / 1024) + 1 : second.size  / 1024;
		sub<<<num_blocks, 1024>>>(data, second.data, temp, size);

		return Vector(&temp, second.size);
	}
	return Vector();
}

bool Vector::operator ! (void) {
	// Could've also checked for this->data == NULL
	// We guarantee that size == 0 iff data == NULL. Only occurs for uninitialized Vector.
	return (this->size == 0);
}

void Vector::createSparse(vector<int> &val, int size, int defval) {
	bool flag = false;

	if(find(val.begin(), val.end(), size - 1) == val.end()) {
		val.push_back(size - 1);
		flag = true;
	}

	//thrust::cuda::par_vector temp_indices(val.begin(), val.end());
	int *temp_indices;
	double *temp_values;
	//thrust::cuda::par_vector temp_values(size, 1);

	cudaMalloc((void **) &temp_indices, val.size() * sizeof(int));
	cudaMalloc((void **) &temp_values, val.size() * sizeof(double));
	cudaMalloc((void **) &data, size * sizeof(double));

	cudaMemcpy(temp_indices, &val[0], val.size() * sizeof(int), cudaMemcpyHostToDevice);
	//thrust::fill(thrust::cuda::par, temp_values, temp_values + val.size(), defval);
	int num_blocks = val.size() % 1024 ? (val.size() / 1024) + 1 : val.size() / 1024;
	setter<<<num_blocks, 1024>>>(temp_values, val.size(), defval);
	cusparseDsctr(handle, size,
		temp_values,
		temp_indices,
		data,
		CUSPARSE_INDEX_BASE_ZERO);
	this->size = size;

	if(flag)
		val.pop_back();

	cudaFree(temp_indices);
	cudaFree(temp_values);
}

bool Vector::check_convergence(double bound) {
	return (thrust::count_if(thrust::cuda::par, data, data + size, convergence<double>(bound))) == 0;
}

void Vector::elementwiseInvert(void) {
	int num_blocks = size % 1024 ? (size / 1024) + 1 : size  / 1024;
	invert<<<num_blocks, 1024>>>(data, size);
}

Vector Vector::getDanglingVector(void) {
	double *temp;

	cudaMalloc((void **) &(temp), size * sizeof(double));
	cudaMemset(temp, 0, size * sizeof(double));

	int num_blocks = size % 1024 ? (size / 1024) + 1 : size  / 1024;
	dangling<<<num_blocks, 1024>>>(data, temp, size);

	return Vector(&temp, size);
}

coo_matrix::coo_matrix(const coo_matrix &temp) {
	this->num_rows = temp.num_rows;
	this->num_cols = temp.num_cols;
	this->nnz = temp.nnz;
	this->descr = descr;

	cudaMalloc((void **) &cooVal, temp.nnz * sizeof(double));
	cudaMalloc((void **) &cooRow, temp.nnz * sizeof(int));
	cudaMalloc((void **) &cooCol, temp.nnz * sizeof(int));
	cudaMemcpy(cooVal, temp.cooVal, temp.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cooRow, temp.cooRow, temp.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cooCol, temp.cooCol, temp.nnz * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMalloc((void **) &csrRowPtr, (num_rows + 1) * sizeof(int));
	cusparseXcoo2csr(handle,
		cooRow,
		nnz,
		num_rows,
		csrRowPtr,
		CUSPARSE_INDEX_BASE_ZERO
	);
}

coo_matrix::coo_matrix(vector<int> &row, vector<int> &col, vector<double> &val, int num_rows, int num_cols) {
	//vector<int> temp_row, temp_col;
	//int prev_index = row[0];

	this->num_rows = num_rows;
	this->num_cols = num_cols;
	this->nnz = val.size();

	cudaMalloc((void **) &cooVal, val.size() * sizeof(double));
	cudaMalloc((void **) &cooRow, val.size() * sizeof(int));
	cudaMalloc((void **) &cooCol, val.size() * sizeof(int));
	cudaMemcpy(cooVal, &val[0], val.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cooRow, &row[0], val.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cooCol, &col[0], val.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &csrRowPtr, (num_rows + 1) * sizeof(int));
	cusparseXcoo2csr(handle,
		cooRow,
		nnz,
		num_rows,
		csrRowPtr,
		CUSPARSE_INDEX_BASE_ZERO
	);

	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
}

void coo_matrix::diagonalize(Vector &arr) {
	this->num_rows = this->num_cols = this->nnz = arr.getSize();

	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	cudaMalloc((void **) &cooVal, nnz * sizeof(double));
	cudaMalloc((void **) &cooRow, num_rows * sizeof(int));
	cudaMalloc((void **) &cooCol, num_cols * sizeof(int));

	int num_blocks = nnz % 1024 ? (nnz / 1024) + 1 : nnz / 1024;
	sequence<<<num_blocks, 1024>>>(cooRow, arr.getSize());
	sequence<<<num_blocks, 1024>>>(cooCol, arr.getSize());
	//thrust::sequence(thrust::cuda::par, cooRow, cooRow + arr.getSize());
	//thrust::sequence(thrust::cuda::par, cooCol, cooCol + arr.getSize());
	cudaMemcpy(cooRow, arr.getData(), num_rows * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cooVal, arr.getData(), num_cols * sizeof(double), cudaMemcpyDeviceToDevice);

	cudaMalloc((void **) &csrRowPtr, (num_rows + 1) * sizeof(int));
	cusparseXcoo2csr(handle,
		cooRow,
		nnz,
		num_rows,
		csrRowPtr,
		CUSPARSE_INDEX_BASE_ZERO
	);
}

coo_matrix &coo_matrix::operator * (double val) {
	thrust::transform(thrust::cuda::par, cooVal, cooVal + nnz, cooVal, scaling(val));
	return *this;
}

coo_matrix coo_matrix::operator *(coo_matrix &mat) {
	/* TODO: Srinath, please examine this function. Refer to csrgemm in the documentation. */
	// I've seen the doc, the code below seems to be okay. Will need to run and check.
	if(num_cols == mat.num_rows) {
		cusparseMatDescr_t descr_C;

		cusparseCreateMatDescr(&descr_C);
		cusparseSetMatType(descr_C, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr_C, CUSPARSE_INDEX_BASE_ZERO);

		int base_C, nnz_C;
		int *nnzTotalDevHostPtr = &nnz;
		int *csr, *row, *col;
		double *val;
		// No clue what it does: don't know how to restore it
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
		cudaMalloc((void **) &csr, sizeof(int) * (num_rows + 1));
		cusparseXcsrgemmNnz(handle, num_rows, mat.num_cols, num_cols,
			descr, nnz, csrRowPtr, cooVal,
			mat.descr, mat.nnz, mat.csrRowPtr, mat.cooVal,
			descr_C, csr, nnzTotalDevHostPtr);

		if(nnzTotalDevHostPtr != NULL) {
			nnz_C = *nnzTotalDevHostPtr;
		} else {
			cudaMemcpy(&nnz_C, csr + num_rows, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&base_C, csr, sizeof(int), cudaMemcpyDeviceToHost);
			nnz_C -= base_C;
		}
		cudaMalloc((void **) &col, sizeof(int) * nnz_C);
		cudaMalloc((void **) &val, sizeof(double) * nnz_C);
		cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			num_rows, mat.num_cols, num_cols,
			descr, nnz, cooVal, csrRowPtr, cooCol,
			mat.descr, mat.nnz, mat.cooVal, mat.csrRowPtr, mat.cooCol,
			descr_C, val, csr, col);

		// Get coorow from csr row
		cudaMalloc((void **) &row, sizeof(int) * nnz_C);
		cusparseXcsr2coo(handle, csr, nnz_C, num_rows, row, CUSPARSE_INDEX_BASE_ZERO);

		return coo_matrix(mat.num_cols, num_rows, nnz_C, &val, &row, &col, &csr, descr_C);
	}
	return coo_matrix();
}

Vector coo_matrix::operator * (Vector &arr) {
	if(num_cols != arr.getSize())
		return Vector();

	double alpha = 1.0;

	double *temp;
	cudaMalloc((void **) &temp, num_rows * sizeof(double));
	cout << "Starting multiplication" << endl;
	cudaMemset(temp, 0, num_rows * sizeof(double));
	cusparseDcsrmv(handle,
		    CUSPARSE_OPERATION_NON_TRANSPOSE,
		    num_rows, num_cols, nnz, 
		    &alpha,
		    descr,
		    cooVal,
		    csrRowPtr,
		    cooCol,
		    arr.getData(),
		    &alpha,
		    temp
	);

	return Vector(&temp, num_rows);
}

coo_matrix coo_matrix::operator ~ (void) {
	cusparseMatDescr_t descr;

	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	double *val;
	int *row, *col, *csr;
	cudaMalloc((void **) &val, nnz * sizeof(double));
	cudaMalloc((void **) &row, nnz * sizeof(int));
	cudaMalloc((void **) &col, nnz * sizeof(int));
	cudaMemcpy(val, cooVal, nnz * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(col, cooRow, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(row, cooCol, nnz * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMalloc((void **) &csr, (num_cols + 1) * sizeof(int));
	cusparseXcoo2csr(handle,
		row,
		nnz,
		num_cols,
		csr,
		CUSPARSE_INDEX_BASE_ZERO
	);

	return coo_matrix(num_cols, num_rows, nnz, &val, &row, &col, &csr, descr);
}

ostream &operator << (ostream &out, coo_matrix &mat) {
	double *val;
	int *row, *col;

	val = (double *) malloc(mat.nnz * sizeof(double));
	row = (int *) malloc(mat.nnz * sizeof(int));
	col = (int *) malloc(mat.nnz * sizeof(int));
	cudaMemcpy(val, mat.cooVal, mat.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(col, mat.cooCol, mat.nnz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(row, mat.cooRow, mat.nnz * sizeof(int), cudaMemcpyDeviceToHost);

	out << "< " <<  mat.num_rows << ", " << mat.num_cols << ", " << mat.nnz << " >\n";
	for(int i = 0; i < mat.nnz; i++)
		out << row[i] << "\t" << col[i] << "\t" << val[i] << endl;

	free(val);
	free(row);
	free(col);

	return out;
}

ostream &operator << (ostream &out, Vector &vec) {
	double *val;

	val = (double *) malloc(vec.size * sizeof(double));
	cudaMemcpy(val, vec.data, vec.size * sizeof(double), cudaMemcpyDeviceToHost);
	
	out << "< " <<  vec.size << " >\n";
	for(int i = 0; i < vec.size; i++)
		out << val[i] << endl;

	free(val);

	return out;
}

///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
/*
Functions pertinent to the problem
They are not generic. (related to Vector and Matrix)
*/
///////////////////////////////////////////////////////

coo_matrix getMatrix(vector<int> &vec, int row_size, int col_size) {
	cusparseMatDescr_t descr;

	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	double *val;
	int *row, *col, *csr;
	int num_blocks = vec.size() % 1024 ? (vec.size() / 1024) + 1 : vec.size() / 1024;
	cudaMalloc((void **) &val, vec.size() * sizeof(double));
	cudaMalloc((void **) &row, vec.size() * sizeof(int));
	cudaMalloc((void **) &col, vec.size() * sizeof(int));

	// The next four lines are for creating a dia matrix with ones in the required places
	// i.e. the indices corresponding to words in the query.
	std::sort(vec.begin(), vec.end());
	cudaMemcpy(row, &vec[0], vec.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(col, row, vec.size() * sizeof(int), cudaMemcpyDeviceToDevice);
	setter<<<num_blocks, 1024>>>(val, vec.size(), 1);

	cudaMalloc((void **) &csr, (row_size + 1) * sizeof(int));
	cusparseXcoo2csr(handle,
		row,
		vec.size(),
		row_size,
		csr,
		CUSPARSE_INDEX_BASE_ZERO
	);

	return coo_matrix(col_size, row_size, vec.size(), &val, &row, &col, &csr, descr);
}

Vector getIntersection(coo_matrix &base) {
	double *vec;
	int size = base.getNumRows();

	cudaMalloc((void **) &vec, sizeof(double) * size);
	int num_blocks = size % 1024 ? (size / 1024) + 1 : size / 1024;
	setter<<<num_blocks, 1024>>>(vec, size, 0);
	add_columns<<<num_blocks, 1024>>>(base.cooCol, vec, size);
	check_intersection<<<num_blocks, 1024>>>(vec, base.getNumCols(), size);

	return Vector(&vec, size);
}

Vector getInduced(coo_matrix &mat, Vector &base) {
	Vector inlinks = mat * base;
	Vector outlinks = (~mat) * base;

	// Changing `base', since it's not being used elsewhere. Once induced_urls are generated, `base' is useless.
	base += inlinks + outlinks;

	int num_blocks = base.getSize() % 1024 ? (base.getSize() / 1024) + 1 : base.getSize() / 1024;
	make_binary<<<num_blocks, 1024>>>(induced.getData(), base.getSize());

	return Vector(base);
}
