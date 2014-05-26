#define THRESHOLD	0.000001

coo_matrix readMatrix(char *filename) {

	char buf[64], *ptr;
	int num_rows, num_cols, nnz;

	vector<int> row_indices, col_indices, values;


	FILE *f = fopen(filename, "r");
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
	nnz = atoi(ptr);
	//free(ptr);

	cout << "Read the banner" << endl;
	for (int i = 0; i < nnz; i++) {
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
		else {
			printf("Invalid matrix: number of entries less than %d\n", num_entries);
			return coo_matrix();
		}
	}

	cout << "Parsed the file" << endl;
	if(fgets(buf, sizeof(buf), f) != NULL) {
		printf("Invalid matrix: number of entries greater than %d\n", num_entries);
		return coo_matrix();
	}

	return coo_matrix(row_indices, col_indices, values, num_rows, num_cols);
}

// I thought it is better to keep mat intact.
// If you want to change it and make it in-place do that.
coo_matrix normalize(coo_matrix &mat, Vector &dangling) {
	Vector sum;
	{
		Vector ones(1, mat.getNumCols());

		sum = mat * ones;
	}
	sum.elementwiseInvert();		// in-place
	dangling = sum.getDanglingVector();
	
	return ((~mat) * (coo_matrix().diagonalize(sum)))
}

Vector page_rank(coo_matrix &link, Vector &dangling, double beta) {
	Vector teleport((1  - beta) / double(link.getNumRows()), link.getNumRows());
	Vector prev_rank;
	Vector rank(1, link.getNumRows());

	double beta_n = beta / link.getNumRows();

	do {
		prev_rank = rank;
		rank = ((beta_n) * ((link * rank) + dangling)) + teleport;
	} while((rank - prev_rank).check_convergence(THRSHOLD));

	return Vector(rank);
}

Vector get_base_urls(coo_matrix &doc, vector<int> &query) {
	coo_matrix base = doc * query.getMatrix(doc.getNumRows(), query.size());
	return base.getIntersection();
}

Vector get_induced_urls(coo_matrix &adj, Vector &base_urls) {
	return adj.getInduced(base_urls);
}

Vector get_relevance_vector(coo_matrix &doc, Vector &kwd) {
	return doc * kwd;
}

coo_matrix get_subgraph(coo_matrix &adj, Vector &induced) {
	return adj * (coo_matrix().diagonalize(induced));
}

coo_matrix get_link_matrix(coo_matrix &sub, Vector &relevance) {
	return sub * (coo_matrix().diagonalize(relevance));
}

coo_matrix get_intelligent_mat(coo_matrix &adj, coo_matrix &doc, vector<int> &query, set<int> &urls, Vector &dangling) {

	Vector induced;
	Vector relevance;
	coo_matrix sub;
	/* TODO: Saurabh, the two things (scopes) given below can be executed in parallel. Check if we can exploit this. */
	{
		// Base Urls
		Vector base_urls = get_base_urls(doc, query);
		induced = get_induced_urls(adj, base_urls);
		sub = get_subgraph(adj, induced);
	}
	{
		Vector kwd;
		kwd.createSparse(query, doc.getNumCols(), 1);
		relevance = get_relevance_vector(doc, kwd);
	}
	coo_matrix link = get_link_matrix(sub, relevance);

	return normalize(link, dangling);
}

/* TODO: Saurabh, main has not been written yet. Refer to "search.cu" and try to add openMPI part. */
