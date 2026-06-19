#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"

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

    FILE *fp;
    int colindex, link, i, j = 0, col, c, colmatch = -1, localsum = 0;
    int co, index;

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

    // Initialize PageRank vector
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
    }

    if (rank == MASTER) {
        printf("Initialization complete\n");
    }

    // Read file (all processes read independently)
    fp = fopen(FILEPATH, "r");
    if (fp == NULL) {
        fprintf(stderr, "Rank %d - Errore: impossibile aprire il file '%s'\n", rank, FILEPATH);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // CSC matrix construction with dangling nodes support
    localsum = 0;
    for (i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Rank %d - Errore lettura file alla riga %d\n", rank, i + 1);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        colindex = colindex - 1;
        link = link - 1;
        rowind[i] = link;

        if (i == 0) {
            colmatch = colindex;
            localsum = 1;
        } else if (colmatch == colindex) {
            localsum += 1;
        } else {
            readsum[colmatch] = localsum;
            for (c = colmatch + 1; c <= colindex; c++) {
                colptr[c] = colptr[colmatch] + localsum;
            }
            localsum = 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }
    if (EDGES > 0) {
        readsum[colmatch] = localsum;
        for (c = colmatch + 1; c <= NODES; c++) {
            colptr[c] = EDGES;
        }
    }
    fclose(fp);

    // Column normalization
    index = 0;
    for (i = 0; i < NODES; i++) {
        co = readsum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }

    if (rank == MASTER) {
        printf("val, rowind and colptr have been populated\n");
    }

    // Compute column distribution among processes
    for (i = 0; i < NPROC; i++) {
        if (i == 0) {
            pcols[i] = NODES / NPROC + NODES % NPROC;
            displs_pr[i] = 0;
        } else {
            pcols[i] = NODES / NPROC;
            displs_pr[i] = pcols[i - 1] + displs_pr[i - 1];
        }
    }

    // Compute non-zero element counts and displacements
    j = 0;
    for (i = 0; i < NPROC; i++) {
        j = j + pcols[i];
        int k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = sendcnts[i - 1] + displs[i - 1];
        }
    }

    // Local pointers: each process uses its own portion via pointer arithmetic
    double *rec_val = val + displs[rank];
    int    *rec_row = rowind + displs[rank];
    rec_col = pcols[rank];

    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    double norm = 0.0;
    double dm_local = 0.0, dm_global = 0.0, norm_sq_local = 0.0;

    // Profiling timers
    double t_spmv = 0.0, t_thread_red = 0.0, t_allreduce_dm = 0.0;
    double t_allreduce_prnew = 0.0, t_update = 0.0, t_norm = 0.0;

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
        double t_phase = 0.0;

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
            #pragma omp master
            {
                t_phase = MPI_Wtime();
            }

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

            #pragma omp master
            {
                t_spmv += MPI_Wtime() - t_phase;
            }

            // Block 3: Reduce thread-local sums into process sum array
            #pragma omp master
            {
                t_phase = MPI_Wtime();
            }

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

            #pragma omp master
            {
                t_thread_red += MPI_Wtime() - t_phase;
            }

            // Block 4: Asynchronous MPI allreduce
            MPI_Request request;
            #pragma omp master
            {
                t_phase = MPI_Wtime();
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
                t_allreduce_prnew += MPI_Wtime() - t_phase;
            }
            #pragma omp barrier

            double redistribution = dm_global * DAMPING / NODES;

            // Block 6: Distributed update and local norm computation
            #pragma omp master
            {
                t_phase = MPI_Wtime();
            }

            #pragma omp for reduction(+ : norm_sq_local)
            for (i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;

                prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

                double diff = prnew[global_col] - prold[global_col];
                norm_sq_local += diff * diff;

                prold[global_col] = prnew[global_col];
            }

            #pragma omp master
            {
                t_update += MPI_Wtime() - t_phase;
            }

            // Block 7: Global convergence check
            #pragma omp master
            {
                t_phase = MPI_Wtime();
                double norm_sq_global = 0.0;
                MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                norm = sqrt(norm_sq_global);
                t_norm += MPI_Wtime() - t_phase;
            }
            #pragma omp barrier

        } while (norm > ERROR);

        free(local_sum);
    }
    free(thread_sums);

    // ========================================================================
    // PROFILING
    // ========================================================================
    if (rank == MASTER) {
        double t_total = t_spmv + t_thread_red + t_allreduce_prnew + t_update + t_norm;
        printf("\n--- PROFILO TEMPO (totale su tutte le iterazioni) ---\n");
        printf("  SpMV locale      : %7.3f s  (%5.1f%%)\n", t_spmv, 100.0 * t_spmv / t_total);
        printf("  Riduzione thread : %7.3f s  (%5.1f%%)\n", t_thread_red, 100.0 * t_thread_red / t_total);
        printf("  Allreduce+attesa : %7.3f s  (%5.1f%%)\n", t_allreduce_prnew, 100.0 * t_allreduce_prnew / t_total);
        printf("  Aggiornamento    : %7.3f s  (%5.1f%%)\n", t_update, 100.0 * t_update / t_total);
        printf("  Norma MPI        : %7.3f s  (%5.1f%%)\n", t_norm, 100.0 * t_norm / t_total);
        printf("  Totale misurato  : %7.3f s\n", t_total);
    }

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
        double sum_pr = 0.0;
        for (i = 0; i < NODES; i++) {
            sum_pr += full_pr[i];
        }
        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("=============================================\n");
        printf("Tempo totale power iteration: %f secondi\n", time_spent);
    }

    free(full_pr);

    // Free memory
    free(val); free(rowind); free(colptr); free(readsum);
    free(prold); free(prnew); free(sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);

    MPI_Finalize();
    return 0;
}