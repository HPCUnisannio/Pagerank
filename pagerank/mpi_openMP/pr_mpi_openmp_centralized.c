#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char **argv)
{
    GraphType graph_type = GRAPH_MEDIUM;
    const Graph *graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char *FILEPATH = graph->filepath;

    int NPROC, rank, num_threads;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    int mpi_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK IBRIDO (lettura centralizzata)\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf("=============================================\n\n");
        fflush(stdout);
    }

    int i, j;

    // Data structures: only Master allocates full CSC arrays
    double *val = NULL;
    int *rowind = NULL;

    // Shared graph metadata (all processes)
    int *colptr  = (int *)calloc(NODES + 1, sizeof(int));
    int *readsum = (int *)calloc(NODES, sizeof(int));

    // PageRank vectors
    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)calloc(NODES, sizeof(double));
    double *sum   = (double *)calloc(NODES, sizeof(double));

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

        val    = (double *)calloc(EDGES, sizeof(double));
        rowind = (int *)calloc(EDGES, sizeof(int));

        if (csc_build_from_file(FILEPATH, NODES, EDGES, val, rowind, colptr, readsum) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Column normalization
        csc_normalize_columns(NODES, EDGES, val, colptr, readsum);

        printf("CSC costruita e normalizzata --> Distribuzione dati\n");
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

    double *rec_val = (double *)malloc(my_cnt * sizeof(double));
    int    *rec_row = (int *)malloc(my_cnt * sizeof(int));

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
    double dm_local = 0.0, dm_global = 0.0, norm_sq_local = 0.0;

    // Private sum vectors for each OpenMP thread
    double **thread_sums = (double **)calloc(num_threads, sizeof(double *));

    // ========================================================================
    // COARSE-GRAIN PARALLEL REGION: THREAD POOL CREATED ONCE OUTSIDE DO-WHILE
    // ========================================================================
    #pragma omp parallel private(i, j)
    {
        int tid = omp_get_thread_num();
        thread_sums[tid] = (double *)calloc(NODES, sizeof(double));
        double *local_sum = thread_sums[tid];

        do {
            memset(local_sum, 0, NODES * sizeof(double));

            #pragma omp master
            {
                dm_local = 0.0;
                norm_sq_local = 0.0;
            }

            int global_col_start = displs_pr[rank];

            // Phase A: Parallel sparse matrix-vector multiplication
            int local_col;
            #pragma omp for schedule(dynamic, 512)
            for (local_col = 0; local_col < rec_col; local_col++) {
                int global_col = global_col_start + local_col;
                int start_idx = colptr[global_col] - displs[rank];
                int end_idx = colptr[global_col + 1] - displs[rank];
                for (j = start_idx; j < end_idx; j++) {
                    local_sum[rec_row[j]] += rec_val[j] * prold[global_col];
                }
            }

            // Phase B: Reduce thread-local sums into process sum array
            #pragma omp for
            for (i = 0; i < NODES; i++) {
                double total_row_sum = 0.0;
                int t;
                for (t = 0; t < num_threads; t++) {
                    if (thread_sums[t] != NULL)
                        total_row_sum += thread_sums[t][i];
                }
                sum[i] = total_row_sum;
            }

            // Phase C: Asynchronous MPI reduction with overlap
            MPI_Request request;
            #pragma omp master
            {
                MPI_Iallreduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
            }

            // Compute local dangling mass while network is working
            #pragma omp for reduction(+ : dm_local)
            for (i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;
                if (readsum[global_col] == 0) {
                    dm_local += prold[global_col];
                }
            }

            #pragma omp master
            {
                MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            #pragma omp barrier

            double redistribution = dm_global * DAMPING / NODES;

            // Phase D: Distributed update and local norm computation
            #pragma omp for reduction(+ : norm_sq_local)
            for (i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;

                prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

                double diff = prnew[global_col] - prold[global_col];
                norm_sq_local += diff * diff;

                prold[global_col] = prnew[global_col];
            }

            // Phase E: Global convergence check
            #pragma omp master
            {
                double norm_sq_global = 0.0;
                MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                norm = sqrt(norm_sq_global);
            }
            #pragma omp barrier

        } while (norm > ERROR);

        free(local_sum);
    }
    free(thread_sums);

    // ========================================================================
    // PROFILING
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;

    // ========================================================================
    // FINAL GATHERING AND VALIDATION
    // ========================================================================
    double *full_pr = (double *)malloc(NODES * sizeof(double));

    MPI_Allgatherv(prnew + displs_pr[rank], pcols[rank], MPI_DOUBLE,
                   full_pr, pcols, displs_pr, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    if (rank == MASTER) {
        double sum_pr = pagerank_validate(NODES, full_pr);
        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("Tempo iterazioni: %f secondi\n", time_spent);

        /*
        for(i = 0; i < NODES; i++) {
            printf("Nodo %d: PageRank = %.10f\n", i, full_pr[i]);
        }
        */
    }

    free(full_pr);

    // Free memory
    free(colptr); free(readsum);
    free(prold); free(prnew); free(sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);
    free(rec_val); free(rec_row);

    MPI_Finalize();
    return 0;
}