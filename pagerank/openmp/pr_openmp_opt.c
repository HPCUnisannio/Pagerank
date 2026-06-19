#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"

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

    FILE *fp;
    int colindex, link, i, j = 0, col, colmatch = 0, localsum = 0;
    int co, index, t;

    double *val    = (double *) calloc(EDGES,       sizeof(double));
    int    *rowind = (int *)    calloc(EDGES,        sizeof(int));
    int    *colptr = (int *)    calloc(NODES + 1,    sizeof(int));

    double *prold = (double *) malloc(NODES * sizeof(double));
    double *prnew = (double *) calloc(NODES,  sizeof(double));
    double *damp1 = (double *) malloc(NODES * sizeof(double));
    double *damp2 = (double *) malloc(NODES * sizeof(double));
    double *diff  = (double *) calloc(NODES,  sizeof(double));

    double norm, norm_sq;

    int *sum = (int *) calloc(NODES, sizeof(int));

    if (!val || !rowind || !colptr || !prold || !prnew ||
        !damp1 || !damp2 || !diff || !sum) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        return 1;
    }

    // Initialize vectors
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = DAMPING;
        damp2[i] = (1.0 - DAMPING) / NODES;
    }
    printf("initialization complete\n");

    const char *filename = FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        return 1;
    }

    // CSC matrix construction from edge list
    for (i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Errore lettura file alla riga %d\n", i + 1);
            fclose(fp);
            return 1;
        }

        colindex = colindex - 1;
        link     = link - 1;
        rowind[i] = link;

        if (colmatch == colindex) {
            localsum += 1;
        } else {
            sum[j]       = localsum;
            colptr[j + 1] = colptr[j] + localsum;
            localsum     = 1;
            j           += 1;
            colmatch     = colindex;
        }

        val[i] = 1.0;
    }

    sum[j]      = localsum;
    colptr[j + 1] = EDGES;
    fclose(fp);

    // Column normalization
    index = 0;
    for (i = 0; i < NODES; i++) {
        co = sum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }

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
        int i_local, j_local, col_local, t_local;

        do {
            // Clear private array
            for (i_local = 0; i_local < NODES; i_local++) {
                local_prnew[tid][i_local] = 0.0;
            }

            // Zero-atomic SpMV: each thread writes to its own private array
            #pragma omp for
            for (col_local = 0; col_local < NODES; col_local++) {
                for (j_local = colptr[col_local]; j_local < colptr[col_local + 1]; j_local++) {
                    local_prnew[tid][rowind[j_local]] += val[j_local] * prold[col_local];
                }
            }

            // Reset norm_sq
            #pragma omp single nowait
            {
                norm_sq = 0.0;
            }

            // Reduction: merge private arrays, apply damping, compute norm
            #pragma omp for reduction(+:norm_sq)
            for (i_local = 0; i_local < NODES; i_local++) {
                double sum_local = 0.0;
                for (t_local = 0; t_local < num_threads; t_local++) {
                    sum_local += local_prnew[t_local][i_local];
                }

                prnew[i_local] = sum_local * damp1[i_local] + damp2[i_local];

                diff[i_local]  = prnew[i_local] - prold[i_local];
                norm_sq += diff[i_local] * diff[i_local];
                prold[i_local] = prnew[i_local];
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
    double sum_pr = 0.0;
    for (i = 0; i < NODES; i++) sum_pr += prnew[i];
    printf("Somma PR = %.10f\n", sum_pr);

    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(damp1);
    free(damp2);
    free(diff);
    free(sum);

    for (t = 0; t < num_threads; t++) {
        free(local_prnew[t]);
    }
    free(local_prnew);

    return 0;
}