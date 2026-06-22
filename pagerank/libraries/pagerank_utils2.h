#ifndef PAGERANK_UTILS2_H
#define PAGERANK_UTILS2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int N;          // number of nodes
    int E;          // number of edges
    int *row_ptr;   // row pointers (size N+1)
    int *col_idx;   // column indices (size E)
    double *val;    // values (size E)
    int *outdeg;    // outdegree (size N)
} CSRGraph;

int csr_build_from_file(const char *filepath, CSRGraph *g);
void csr_normalize(CSRGraph *g);
void csr_free(CSRGraph *g);

#endif