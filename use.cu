#include <iostream>
#include <cstdio>
#include <vector>

#include <cstring>
#include <cstdlib>
#include <cerrno>

#include "coo_mat.h"

using namespace std;

int main() {
	vector<int> row_indices, col_indices;
	vector<double> values;

	int num_rows, num_cols, num_entries;
	char buf[256], *ptr;

	init();
	FILE *f = fopen("matrix_market_test.mtx", "r");
	if (f == NULL) {
		perror("fopen");
		exit(errno);
	}

	// Reading and discarding the first two lines in the banner, taking the values in the third line
	fgets(buf, sizeof(buf), f);
	fgets(buf, sizeof(buf), f);
	fgets(buf, sizeof(buf), f);
	ptr = strtok(buf, "\t");
	num_rows = atoi(ptr);
	ptr = strtok(NULL, "\t");
	num_cols = atoi(ptr);
	ptr = strtok(NULL, "\t");
	num_entries = atoi(ptr);
	//free(ptr);

	cout << "Read the banner" << endl;
	for (int i = 0; i < num_entries; i++) {
		cout << "Line number:" << i << endl;
		if(fgets(buf, sizeof(buf), f) != NULL) {
			ptr = strtok(buf, "\t");
			row_indices.push_back(atoi(ptr) - 1);

			ptr = strtok(NULL, "\t");
			col_indices.push_back(atoi(ptr) - 1);

			ptr = strtok(NULL, "\t");
			values.push_back(atoi(ptr));

			//free(ptr);
		}
		else
			printf("Invalid matrix: number of entries less than %d\n", num_entries);
	}

	cout << "Parsed the file" << endl;
	if(fgets(buf, sizeof(buf), f) != NULL) {
		printf("Invalid matrix: number of entries greater than %d\n", num_entries);
	}

	vector<double> vec;
	vec.push_back(1);
	vec.push_back(1);

	Vector v(vec);
	coo_matrix c(row_indices, col_indices, values, num_rows, num_cols);
	cout << c << endl;
	cout << v << endl;

	Vector t = ((~c) * v) * 3;
	Vector t_dash = t + t;

	coo_matrix n;
	n.diagonalize(t_dash);
	cout << n << endl;

	fclose(f);
}
