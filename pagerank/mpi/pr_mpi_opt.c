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

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    if (rank == MASTER) {
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK (PURE MPI - I/O Centralizzato)\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali : %d\n", NPROC);
        printf("=============================================\n\n");
        fflush(stdout);
    }

    int i, j;

    // Data structures: only Master allocates full CSC arrays
    double *val = NULL;
    int *rowind = NULL;

    // Shared graph metadata
    int *colptr  = (int*)calloc(NODES + 1, sizeof(int));
    int *readsum = (int*)calloc(NODES, sizeof(int));

    // PageRank vectors
    double *prold     = (double*)malloc(NODES * sizeof(double));
    double *prnew     = (double*)calloc(NODES, sizeof(double));
    double *local_sum = (double*)calloc(NODES, sizeof(double));

    // Scalar damping constants
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    // Partitioning arrays
    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *pcols     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_col;

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);

    // ========================================================================
    // 1. FILE READING AND CSC CONSTRUCTION (MASTER ONLY)
    // ========================================================================
    if (rank == MASTER) {
        val    = (double*)calloc(EDGES, sizeof(double));
        rowind = (int*)calloc(EDGES, sizeof(int));

        if (csc_build_from_file(FILEPATH, NODES, EDGES, val, rowind, colptr, readsum) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Column normalization
        csc_normalize_columns(NODES, EDGES, val, colptr, readsum);

        printf("CSC costruita e normalizzata --> Inizio Distribuzione Rete\n");
    }

    // ========================================================================
    // 2. DISTRIBUTE GRAPH METADATA AND CSC PORTIONS
    // ========================================================================
    MPI_Bcast(colptr, NODES + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(readsum, NODES, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Compute column distribution
    compute_column_distribution(NODES, NPROC, pcols, displs_pr);

    // Compute non-zero element counts and displacements
    compute_nnz_distribution(NPROC, pcols, colptr, sendcnts, displs);

    int my_cnt = sendcnts[rank];
    rec_col = pcols[rank];

    double *rec_val = (double*)malloc(my_cnt * sizeof(double));
    int    *rec_row = (int*)malloc(my_cnt * sizeof(int));

    // Master distributes matrix portions to workers
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Master frees full arrays after distribution
    if (rank == MASTER) {
        free(val);
        free(rowind);
    }

    // ========================================================================
    // 3. POWER ITERATION CORE
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();
    double norm = 0.0;

    double dm_local, dm_global, norm_sq_local, norm_sq_global;

    do {
        memset(local_sum, 0, NODES * sizeof(double));
        dm_local      = 0.0;
        norm_sq_local = 0.0;

        int global_col_start = displs_pr[rank];

        // Phase A: Local sparse matrix-vector multiplication
        for (int local_col = 0; local_col < rec_col; local_col++) {
            int global_col = global_col_start + local_col;
            int start_idx  = colptr[global_col] - displs[rank];
            int end_idx    = colptr[global_col + 1] - displs[rank];

            for (j = start_idx; j < end_idx; j++) {
                local_sum[rec_row[j]] += rec_val[j] * prold[global_col];
            }
        }

        // Phase B: Asynchronous network and overlap with dangling mass computation
        MPI_Request request;
        MPI_Iallreduce(local_sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

        // Compute local dangling mass while reduction is in progress
        dm_local = pagerank_compute_dangling_mass_range(prold, readsum,
                                                        global_col_start,
                                                        global_col_start + rec_col);

        // Synchronize dangling mass and wait for prnew to be ready
        MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        double redistribution = dm_global * DAMP1 / NODES;

        // Phase C: Symmetric distributed post-processing
        pagerank_update_and_norm_range(prnew, prold,
                                       global_col_start, global_col_start + rec_col,
                                       redistribution, DAMPING, NODES, &norm_sq_local);

        // Phase D: Convergence evaluation
        MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm = sqrt(norm_sq_global);

    } while (norm > ERROR);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;

    // ========================================================================
    // 4. FINAL GATHERING AND VALIDATION
    // ========================================================================
    double *full_pr = (double*)malloc(NODES * sizeof(double));

    MPI_Allgatherv(prold + displs_pr[rank], pcols[rank], MPI_DOUBLE,
                   full_pr, pcols, displs_pr, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    if (rank == MASTER) {
        double sum_pr = pagerank_validate(NODES, full_pr);

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("Tempo puro di Iterazione: %f secondi\n", time_spent);
        printf("=============================================\n");

        /*
        for(i = 0; i < NODES; i++) {
            printf("Node %d: PageRank = %.10f\n", i, full_pr[i]);
        }
        */
    }

    // Free memory
    free(full_pr);
    free(colptr); free(readsum);
    free(prold); free(prnew); free(local_sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);
    free(rec_val); free(rec_row);

    MPI_Finalize();
    return 0;
}