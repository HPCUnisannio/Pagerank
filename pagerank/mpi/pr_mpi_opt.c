#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"

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

    int i, j, c;
    int colindex, link, colmatch = -1, localsum = 0;
    int co, index;

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

    // Scalar damping constants (replaces dense arrays damp1, damp2)
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    // Partitioning arrays
    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *pcols     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_col;

    // Initialize PageRank vector
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
    }

    // ========================================================================
    // 1. FILE READING AND CSC CONSTRUCTION (MASTER ONLY)
    // ========================================================================
    if (rank == MASTER) {
        FILE *fp = fopen(FILEPATH, "r");
        if (fp == NULL) {
            fprintf(stderr, "Rank %d - Errore: impossibile aprire il file '%s'\n", rank, FILEPATH);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        val    = (double*)calloc(EDGES, sizeof(double));
        rowind = (int*)calloc(EDGES, sizeof(int));

        localsum = 0;
        for (i = 0; i < EDGES; i++) {
            if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
                fprintf(stderr, "Rank %d - Errore lettura file\n", rank);
                fclose(fp);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            colindex--; link--;
            rowind[i] = link;

            if (i == 0) {
                colmatch = colindex;
                localsum = 1;
            } else if (colmatch == colindex) {
                localsum++;
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
            for (j = index; j < index + co; j++) val[j] /= co;
            index += co;
        }
        printf("CSC costruita e normalizzata --> Inizio Distribuzione Rete\n");
    }

    // ========================================================================
    // 2. DISTRIBUTE GRAPH METADATA AND CSC PORTIONS
    // ========================================================================
    MPI_Bcast(colptr, NODES + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(readsum, NODES, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Compute column distribution
    for (i = 0; i < NPROC; i++) {
        if (i == 0) {
            pcols[i]      = NODES / NPROC + NODES % NPROC;
            displs_pr[i]  = 0;
        } else {
            pcols[i]      = NODES / NPROC;
            displs_pr[i]  = pcols[i - 1] + displs_pr[i - 1];
        }
    }

    // Compute non-zero element counts and displacements
    j = 0;
    for (i = 0; i < NPROC; i++) {
        j += pcols[i];
        int k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0) displs[i] = 0;
        else        displs[i] = sendcnts[i - 1] + displs[i - 1];
    }

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
        dm_local     = 0.0;
        norm_sq_local = 0.0;

        int global_col_start = displs_pr[rank];

        // Phase A: Local sparse matrix-vector multiplication
        int local_col;
        for (local_col = 0; local_col < rec_col; local_col++) {
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
        for (local_col = 0; local_col < rec_col; local_col++) {
            int global_col = global_col_start + local_col;
            if (readsum[global_col] == 0) {
                dm_local += prold[global_col];
            }
        }

        // Synchronize dangling mass and wait for prnew to be ready
        MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        double redistribution = dm_global * DAMP1 / NODES;

        // Phase C: Symmetric distributed post-processing
        for (local_col = 0; local_col < rec_col; local_col++) {
            int global_col = global_col_start + local_col;

            prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

            double diff = prnew[global_col] - prold[global_col];
            norm_sq_local += diff * diff;

            prold[global_col] = prnew[global_col];
        }

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
        double sum_pr = 0;
        for (i = 0; i < NODES; i++) sum_pr += full_pr[i];

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("Tempo puro di Iterazione: %f secondi\n", time_spent);
        printf("=============================================\n");
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