#define THRESHOLD	0.000001


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
