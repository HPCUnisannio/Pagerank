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

    if (rank == 0) {
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK IBRIDO [ MPI(post processing distribuito) + OpenMP(grana grossa + privatizzazione) ]\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf("=============================================\n\n");
        fflush(stdout);
    }

    int i, j;

    // Allocate main structures (replicated on all processes)
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

    int rec_col;

    // ========================================================================
    // PHASE 1: SETUP (Master measures, all synchronized at end)
    // ========================================================================
    double t_setup_start, t_setup_end, setup_time;

    if (rank == MASTER) {
        t_setup_start = MPI_Wtime();
    }

    // Build CSC matrix from file (all processes read independently)
    if (csc_build_from_file(FILEPATH, NODES, EDGES, val, rowind, colptr, readsum) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Normalize columns
    csc_normalize_columns(NODES, EDGES, val, colptr, readsum);

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);

    // Compute column distribution among processes
    compute_column_distribution(NODES, NPROC, pcols, displs_pr);

    // Compute non-zero element counts and displacements
    compute_nnz_distribution(NPROC, pcols, colptr, sendcnts, displs);

    // Local pointers: each process uses its own portion via pointer arithmetic
    double *rec_val = val + displs[rank];
    int    *rec_row = rowind + displs[rank];
    rec_col = pcols[rank];

    // Synchronize: ensure ALL processes have completed setup
    MPI_Barrier(MPI_COMM_WORLD);

    double t_compute_start, t_compute_end, compute_time;
    if (rank == MASTER) {
        t_setup_end = MPI_Wtime();
        setup_time = t_setup_end - t_setup_start;
        printf("Initialization complete\n");
        printf("val, rowind and colptr have been populated\n");
        t_compute_start = t_setup_end;
    }

    // ========================================================================
    // PHASE 2: COMPUTATION (Master measures, all synchronized at both ends)
    // ========================================================================
    
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
            // Block 1: Reset local arrays and accumulators
            memset(local_sum, 0, NODES * sizeof(double));

            #pragma omp master
            {
                dm_local = 0.0;
                norm_sq_local = 0.0;
            }

            int global_col_start = displs_pr[rank];

            // Block 2: Parallel sparse matrix-vector multiplication
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

            // Block 3: Reduce thread-local sums into process sum array
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

            // Block 4: Asynchronous MPI allreduce
            MPI_Request request;
            #pragma omp master
            {
                MPI_Iallreduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
            }

            // Block 5: Overlap - compute local dangling mass while network works
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

            // Block 6: Distributed update and local norm computation
            #pragma omp for reduction(+ : norm_sq_local)
            for (i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;

                prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

                double diff = prnew[global_col] - prold[global_col];
                norm_sq_local += diff * diff;

                prold[global_col] = prnew[global_col];
            }

            // Block 7: Global convergence check
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

    // Synchronize: ensure ALL processes have exited the loop
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        t_compute_end = MPI_Wtime();
        compute_time = t_compute_end - t_compute_start;
    }

    // ========================================================================
    // PHASE 3: FINAL GATHERING AND VALIDATION
    // ========================================================================
    double *full_pr = (double *)malloc(NODES * sizeof(double));

    MPI_Allgatherv(prnew + displs_pr[rank], pcols[rank], MPI_DOUBLE,
                   full_pr, pcols, displs_pr, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    if (rank == MASTER) {
        double sum_pr = pagerank_validate(NODES, full_pr);
        printf("=============================================\n");
        printf("VERIFICA MATEMATICA: Somma finale PR = %.10f\n", sum_pr);
        printf("=============================================\n");

        // Print execution summary
        char label[60];
        sprintf(label, "MPI+OpenMP REPL (%d proc, %d thr)", NPROC, num_threads);
        double total_time = setup_time + compute_time;
        measure_print_summary(label, setup_time, compute_time, total_time);
  
        // Read sequential time for comparison
        double sequential_time = 0.0;
        FILE *time_file = fopen("sequential/sequential_time.txt", "r");
        if (time_file) {
            fscanf(time_file, "%lf", &sequential_time);
            fclose(time_file);
            
            char label1[30];
            sprintf(label1, "Sequenziale");
            char label2[50];
            sprintf(label2, "MPI+OpenMP REPL (%d proc, %d thr)", NPROC, num_threads);
            
            measure_print_comparison(label1, sequential_time, label2, compute_time);
        } else {
            printf("Run sequential version first to generate the baseline.\n");
        }

        /*
        for(i = 0; i < NODES; i++) {
            printf("Nodo %d: PageRank = %.10f\n", i, full_pr[i]);
        }
        */
    }

    free(full_pr);

    // Free memory
    free(val); free(rowind); free(colptr); free(readsum);
    free(prold); free(prnew); free(sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);

    MPI_Finalize();
    return 0;
}