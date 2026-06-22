#include "pagerank_utils2.h"

int csr_build_from_file(const char *filepath, CSRGraph *g)
{
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        printf("ERROR: cannot open %s\n", filepath);
        return -1;
    }

    int src, dst;
    int max_node = 0;
    int edges = 0;
    int i = 0;

    /* PASS 1: count */
    while (fscanf(fp, "%d %d", &src, &dst) == 2) {
        if (src > max_node) max_node = src;
        if (dst > max_node) max_node = dst;
        edges++;
    }

    rewind(fp);

    g->N = max_node;   // or max_node + 1 if 0-based needed
    g->E = edges;

    printf("DEBUG CSR: N=%d E=%d\n", g->N, g->E);

    g->outdeg = calloc(g->N, sizeof(int));
    int *srcs = malloc(sizeof(int) * edges);
    int *dsts = malloc(sizeof(int) * edges);

    for (i = 0; i < edges; i++) {
        fscanf(fp, "%d %d", &srcs[i], &dsts[i]);
        srcs[i]--;
        dsts[i]--;
        g->outdeg[srcs[i]]++;
    }

    fclose(fp);

    g->row_ptr = malloc((g->N + 1) * sizeof(int));
    g->row_ptr[0] = 0;

    for (i = 0; i < g->N; i++) {
        g->row_ptr[i + 1] = g->row_ptr[i] + g->outdeg[i];
    }

    int *pos = calloc(g->N, sizeof(int));

    g->col_idx = malloc(edges * sizeof(int));
    g->val     = malloc(edges * sizeof(double));

    for (i = 0; i < edges; i++) {
        int s = srcs[i];
        int d = dsts[i];

        int idx = g->row_ptr[s] + pos[s]++;
        g->col_idx[idx] = d;
        g->val[idx] = 1.0;
    }

    free(srcs);
    free(dsts);
    free(pos);

    return 0;
}
/* Normalize outgoing links */
void csr_normalize(CSRGraph *g)
{
    int i = 0;
    for (i = 0; i < g->N; i++) {
        int deg = g->outdeg[i];
        if (deg == 0) continue;
        int j;
        for (j = g->row_ptr[i]; j < g->row_ptr[i+1]; j++) {
            g->val[j] /= deg;
        }
    }
}

void csr_free(CSRGraph *g)
{
    free(g->row_ptr);
    free(g->col_idx);
    free(g->val);
    free(g->outdeg);
}