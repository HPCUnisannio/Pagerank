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

    FILE *fp;
    int colindex, link, i, j = 0, col, c, colmatch = -1, localsum = 0;
    int co, index;

    // Allocate main structures
    double *val    = (double *)calloc(EDGES, sizeof(double));
    int    *rowind = (int *)calloc(EDGES, sizeof(int));
    int    *colptr = (int *)calloc(NODES + 1, sizeof(int));
    int    *readsum = (int *)calloc(NODES, sizeof(int));

    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)calloc(NODES, sizeof(double));
    double *sum   = (double *)calloc(NODES, sizeof(double));

    // Scalar damping constants replace dense arrays
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *pcols     = (int *)malloc(NPROC * sizeof(int));
    int *displs_pr = (int *)malloc(NPROC * sizeof(int));

    double norm;
    int rec_col;

    // Initialize PageRank vector
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
    }

    if (rank == MASTER) {
        printf("Initialization complete\n");
    }

    // Read file
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

    // Profiling timers
    double t_scatter = 0.0, t_spmv = 0.0, t_reduce_mpi = 0.0;
    double t_postproc = 0.0, t_bcast = 0.0;

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

                double t0 = MPI_Wtime();
                MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE,
                             rec_pr, pcols[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                t_scatter += MPI_Wtime() - t0;
            }
            #pragma omp barrier

            double t1 = MPI_Wtime();

            // Parallel SpMV with privatized local_sum per thread
            int global_col_start = (rank == MASTER) ? 0 : displs_pr[rank];

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

            t_spmv += MPI_Wtime() - t1;

            // MPI reduction: only master thread participates
            #pragma omp master
            {
                double t2 = MPI_Wtime();
                MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
                t_reduce_mpi += MPI_Wtime() - t2;
            }
            #pragma omp barrier

            // Post-processing: only master process executes
            if (rank == MASTER) {
                double t_post = MPI_Wtime();

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

                t_postproc += MPI_Wtime() - t_post;
            }

            // Broadcast convergence norm
            #pragma omp master
            {
                if (rank == MASTER) {
                    norm = sqrt(norm_sq);
                }
                double t3 = MPI_Wtime();
                MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                t_bcast += MPI_Wtime() - t3;
            }
            #pragma omp barrier

        } while (norm > ERROR);

        #pragma omp master
        if (rank == MASTER) {
            printf("\n--- PROFILO TEMPO (totale su tutte le iterazioni) ---\n");
            printf("  MPI_Scatterv  : %.4f s\n", t_scatter);
            printf("  SpMV+riduzione: %.4f s\n", t_spmv);
            printf("  MPI_Reduce    : %.4f s\n", t_reduce_mpi);
            printf("  Post-proc     : %.4f s\n", t_postproc);
            printf("  MPI_Bcast     : %.4f s\n", t_bcast);
        }

        free(local_sum);
    }
    free(thread_sums);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;

    if (rank == MASTER) {
        double sum_pr = 0;
        for (i = 0; i < NODES; i++)
            sum_pr += prnew[i];

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("=============================================\n");
        printf("Time taken for power iteration solution: %f seconds\n", time_spent);
    }

    // Free memory
    free(val); free(rowind); free(colptr); free(readsum);
    free(prold); free(prnew); free(sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);
    free(rec_val); free(rec_row); free(rec_pr);

    MPI_Finalize();
    return 0;
}