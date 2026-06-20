#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char *argv[])
{
    int NPROC, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("Program start\n");
        printf("Number of processes %d\n", NPROC);
    }

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    int i, j;

    // Allocate main structures
    double *val    = (double*)calloc(EDGES, sizeof(double));
    int    *rowind = (int*)calloc(EDGES, sizeof(int));
    int    *colptr = (int*)calloc(NODES + 1, sizeof(int));
    int    *readsum = (int*)calloc(NODES, sizeof(int));

    double *prold = (double*)malloc(NODES * sizeof(double));
    double *prnew = (double*)calloc(NODES, sizeof(double));
    double *sum   = (double*)calloc(NODES, sizeof(double));

    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *pcols     = (int*)malloc(NPROC * sizeof(int));
    int *displs_pr = (int*)malloc(NPROC * sizeof(int));

    double norm;
    int rec_col;

    // Build CSC matrix from file (all processes read)
    if (csc_build_from_file(FILEPATH, NODES, EDGES, val, rowind, colptr, readsum) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Normalize columns
    csc_normalize_columns(NODES, EDGES, val, colptr, readsum);

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);

    if (rank == MASTER) {
        printf("Initialization complete\n");
        printf("val, rowind and colptr have been populated\n");
    }

    // Compute column distribution among processes
    compute_column_distribution(NODES, NPROC, pcols, displs_pr);

    // Compute non-zero element counts and displacements
    compute_nnz_distribution(NPROC, pcols, colptr, sendcnts, displs);

    // Allocate local receive buffers
    double *rec_val = (double*)malloc(sendcnts[rank] * sizeof(double));
    int    *rec_row = (int*)malloc(sendcnts[rank] * sizeof(int));
    double *rec_pr  = (double*)malloc(pcols[rank] * sizeof(double));

    // Distribute data
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT, &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    do {
        memset(sum, 0, NODES * sizeof(double));
        if (rank == MASTER) {
            memset(prnew, 0, NODES * sizeof(double));
        }
        norm = 0.0;

        // Distribute PageRank vector portions
        MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE, rec_pr, pcols[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Local sparse matrix-vector multiplication
        int global_col_start = displs_pr[rank];

        for (int local_col = 0; local_col < rec_col; local_col++) {
            int global_col = global_col_start + local_col;
            int start_idx  = colptr[global_col] - displs[rank];
            int end_idx    = colptr[global_col + 1] - displs[rank];

            for (j = start_idx; j < end_idx; j++) {
                sum[rec_row[j]] += rec_val[j] * rec_pr[local_col];
            }
        }

        // Reduce partial results to master
        MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

        // Master applies damping, dangling nodes redistribution, and computes convergence
        if (rank == MASTER) {
            double dangling_mass = pagerank_compute_dangling_mass(NODES, prold, readsum);
            double redistribution = (dangling_mass * DAMPING) / NODES;

            double norm_sq = 0.0;
            pagerank_update_and_norm_range(prnew, prold, 0, NODES,
                                           redistribution, DAMPING, NODES, &norm_sq);
            norm = sqrt(norm_sq);
        }

        // Broadcast norm to all processes for loop continuation decision
        MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    } while (norm > ERROR);

    double end = MPI_Wtime();
    double time_spent = end - begin;

    if (rank == MASTER) {
        double sum_pr = pagerank_validate(NODES, prnew);

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("=============================================\n");
        printf("Time taken for power iteration solution: %f seconds\n", time_spent);

        /*
        for( i = 0; i < NODES; i++) {
            printf("Node %d: PageRank = %.10f\n", i, prnew[i]);
        }
        */
    }

    // Free memory
    free(val); free(rowind); free(colptr); free(readsum);
    free(prold); free(prnew); free(sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);
    free(rec_val); free(rec_row); free(rec_pr);

    MPI_Finalize();
    return 0;
}