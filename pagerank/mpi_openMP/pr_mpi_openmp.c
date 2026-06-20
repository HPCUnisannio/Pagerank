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
    int NPROC, rank, num_threads;

    // Parse thread count from command line, default to max available
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

    if (rank == 0) {
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK IBRIDO [ MPI + OpenMP ]\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf("=============================================\n\n");
        fflush(stdout);
    }

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph *graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char *FILEPATH = graph->filepath;

    int i, j;

    // Allocate main structures
    double *val    = (double *)calloc(EDGES, sizeof(double));
    int    *rowind = (int *)calloc(EDGES, sizeof(int));
    int    *colptr = (int *)calloc(NODES + 1, sizeof(int));
    int    *readsum = (int *)calloc(NODES, sizeof(int));

    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)calloc(NODES, sizeof(double));
    double *sum   = (double *)calloc(NODES, sizeof(double));

    // Scalar damping constants
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *pcols     = (int *)malloc(NPROC * sizeof(int));
    int *displs_pr = (int *)malloc(NPROC * sizeof(int));

    double norm;
    int rec_col;

    // Build CSC matrix from file
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
    double *rec_val = (double *)malloc(sendcnts[rank] * sizeof(double));
    int    *rec_row = (int *)malloc(sendcnts[rank] * sizeof(int));
    double *rec_pr  = (double *)malloc(pcols[rank] * sizeof(double));

    // Distribute data
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE,
                 rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT,
                &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT,
                 rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    double dangling_mass = 0.0;
    double norm_sq = 0.0;

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

            // MPI communication: only master thread participates
            #pragma omp master
            {
                if (rank == MASTER) {
                    memset(prnew, 0, NODES * sizeof(double));
                    dangling_mass = 0.0;
                    norm_sq = 0.0;
                }
                norm = 0.0;

                MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE,
                             rec_pr, pcols[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
            }
            #pragma omp barrier

            // Parallel SpMV with privatized local_sum per thread
            int global_col_start = displs_pr[rank];

            int local_col;
            #pragma omp for schedule(dynamic, 512)
            for (local_col = 0; local_col < rec_col; local_col++) {
                int global_col = global_col_start + local_col;
                int start_idx = colptr[global_col] - displs[rank];
                int end_idx = colptr[global_col + 1] - displs[rank];

                for (j = start_idx; j < end_idx; j++) {
                    local_sum[rec_row[j]] += rec_val[j] * rec_pr[local_col];
                }
            }

            // Reduce thread-local sums into process sum array
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

            // MPI reduction: only master thread participates
            #pragma omp master
            {
                MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
            }
            #pragma omp barrier

            // Post-processing: only master process executes
            if (rank == MASTER) {

                #pragma omp for reduction(+ : dangling_mass)
                for (i = 0; i < NODES; i++) {
                    if (readsum[i] == 0) {
                        dangling_mass += prold[i];
                    }
                }

                double redistribution = dangling_mass * DAMPING / NODES;

                // Fused loop: apply damping, compute diff, update prold
                #pragma omp for reduction(+ : norm_sq)
                for (i = 0; i < NODES; i++) {
                    prnew[i] = prnew[i] * DAMP1 + DAMP2 + redistribution;
                    double diff = prnew[i] - prold[i];
                    norm_sq += diff * diff;
                    prold[i] = prnew[i];
                }
            }

            // Broadcast convergence norm
            #pragma omp master
            {
                if (rank == MASTER) {
                    norm = sqrt(norm_sq);
                }
                MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
            }
            #pragma omp barrier

        } while (norm > ERROR);

        free(local_sum);
    }
    free(thread_sums);

    MPI_Barrier(MPI_COMM_WORLD);
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
        for(i = 0; i < NODES; i++) {
            printf("Nodo %d: PageRank = %.10f\n", i, prnew[i]);
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