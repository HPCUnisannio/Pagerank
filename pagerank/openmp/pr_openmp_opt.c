#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define DAMPING 0.85
#define err     0.00001

double sqrt(double x);

int main(int argc, char *argv[])
{
    printf("Program start\n");

    // Parse thread count from command line, default 4
    int num_threads = 4;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) num_threads = 1;
    }
    omp_set_num_threads(num_threads);
    printf("Using %d threads\n", num_threads);

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    int i, t;

    double *val    = (double *) calloc(EDGES, sizeof(double));
    int    *rowind = (int *)    calloc(EDGES, sizeof(int));
    int    *colptr = (int *)    calloc(NODES + 1, sizeof(int));
    int    *sum    = (int *)    calloc(NODES, sizeof(int));

    double *prold = (double *) malloc(NODES * sizeof(double));
    double *prnew = (double *) calloc(NODES, sizeof(double));

    double norm, norm_sq;

    if (!val || !rowind || !colptr || !prold || !prnew || !sum) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        return 1;
    }

    // Build CSC matrix from file
    const char *filename = FILEPATH;
    if (csc_build_from_file(filename, NODES, EDGES, val, rowind, colptr, sum) != 0) {
        return 1;
    }

    // Normalize columns
    csc_normalize_columns(NODES, EDGES, val, colptr, sum);

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);

    printf("initialization complete\n");
    printf("CSC construction complete\n");

    // Allocate privatized memory: one array per thread
    num_threads = omp_get_max_threads();
    double **local_prnew = (double **)malloc(num_threads * sizeof(double *));
    for (t = 0; t < num_threads; t++) {
        local_prnew[t] = (double *)calloc(NODES, sizeof(double));
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int i_local, col_local;

        do {
            // Clear private array
            memset(local_prnew[tid], 0, NODES * sizeof(double));

            // Zero-atomic SpMV: each thread writes to its own private array
            #pragma omp for
            for (col_local = 0; col_local < NODES; col_local++) {
                int start_idx = colptr[col_local];
                int end_idx   = colptr[col_local + 1];
                for (int j_local = start_idx; j_local < end_idx; j_local++) {
                    local_prnew[tid][rowind[j_local]] += val[j_local] * prold[col_local];
                }
            }

            // Reset norm_sq
            #pragma omp single nowait
            {
                norm_sq = 0.0;
            }

            // Reduction: merge private arrays and apply damping
            #pragma omp for reduction(+:norm_sq)
            for (i_local = 0; i_local < NODES; i_local++) {
                // Merge thread-private contributions
                double sum_local = 0.0;
                for (int t_local = 0; t_local < num_threads; t_local++) {
                    sum_local += local_prnew[t_local][i_local];
                }

                prnew[i_local] = sum_local;

                // Apply damping, compute diff, and update prold inline
                double damped = prnew[i_local] * DAMPING + (1.0 - DAMPING) / NODES;
                double diff = damped - prold[i_local];
                norm_sq += diff * diff;
                prold[i_local] = damped;
            }

            // Compute final norm
            #pragma omp single
            {
                norm = sqrt(norm_sq);
            }

        } while (norm > err);
    }

    double end_time = omp_get_wtime();
    double tempo_parallelo = end_time - start_time;

    printf("\n=============================================\n");
    printf("TEMPO POWER ITERATION OpenMP (PRIVATIZZATA): %.6f secondi\n", tempo_parallelo);
    printf("=============================================\n");

    // Validate PageRank sum equals 1.0
    double sum_pr = pagerank_validate(NODES, prold);
    printf("Somma PR = %.10f\n", sum_pr);

    /*
    for(i = 0; i < NODES; i++) {
        printf("Node %d: PageRank = %.10f\n", i, prnew[i]);
    }
    */

    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(sum);

    for (t = 0; t < num_threads; t++) {
        free(local_prnew[t]);
    }
    free(local_prnew);

    return 0;
}