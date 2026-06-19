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

    if (rank == MASTER) {
        printf("Program start\n");
        printf("Number of processes %d\n", NPROC);
    }

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    FILE *fp;
    int colindex, link, i, j = 0, k, col, c, colmatch = -1, localsum = 0;
    int co, index;

    // Allocate main structures
    double *val    = (double*)calloc(EDGES, sizeof(double));
    int    *rowind = (int*)calloc(EDGES, sizeof(int));
    int    *colptr = (int*)calloc(NODES + 1, sizeof(int));
    int    *readsum = (int*)calloc(NODES, sizeof(int));

    double *prold = (double*)malloc(NODES * sizeof(double));
    double *prnew = (double*)calloc(NODES, sizeof(double));
    double *damp1 = (double*)malloc(NODES * sizeof(double));
    double *damp2 = (double*)malloc(NODES * sizeof(double));
    double *diff  = (double*)calloc(NODES, sizeof(double));
    double *sum   = (double*)calloc(NODES, sizeof(double));

    int *sendcnts   = malloc(sizeof(int) * NPROC);
    int *displs     = malloc(sizeof(int) * NPROC);
    int *pcols      = (int*)malloc(NPROC * sizeof(int));
    int *displs_pr  = (int*)malloc(NPROC * sizeof(int));

    double norm;
    int rec_col;

    // Initialize vectors
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = DAMPING;
        damp2[i] = (1.0 - DAMPING) / NODES;
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
        link     = link - 1;
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

    // Compute non-zero element counts and displacements for irregular distribution
    j = 0;
    for (i = 0; i < NPROC; i++) {
        j = j + pcols[i];
        k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = sendcnts[i - 1] + displs[i - 1];
        }
    }

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
        int global_col_start = (rank == MASTER) ? 0 : displs_pr[rank];

        int local_col;
        for (local_col = 0; local_col < rec_col; local_col++) {
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
            double dangling_mass = 0.0;
            for (i = 0; i < NODES; i++) {
                if (readsum[i] == 0) {
                    dangling_mass += prold[i];
                }
            }

            double redistribution = (dangling_mass * DAMPING) / NODES;

            for (i = 0; i < NODES; i++) {
                prnew[i] = prnew[i] * damp1[i] + damp2[i] + redistribution;
            }

            double norm_sq = 0.0;
            for (i = 0; i < NODES; i++) {
                diff[i] = prnew[i] - prold[i];
                norm_sq += diff[i] * diff[i];
                prold[i] = prnew[i];
            }
            norm = sqrt(norm_sq);
        }

        // Broadcast norm to all processes for loop continuation decision
        MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    } while (norm > ERROR);

    double end = MPI_Wtime();
    double time_spent = end - begin;

    if (rank == MASTER) {
        double sum_pr = 0;
        for (i = 0; i < NODES; i++) sum_pr += prnew[i];

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("=============================================\n");
        printf("Time taken for power iteration solution: %f seconds\n", time_spent);
    }

    // Free memory
    free(val); free(rowind); free(colptr); free(readsum);
    free(prold); free(prnew); free(damp1); free(damp2); free(diff); free(sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);
    free(rec_val); free(rec_row); free(rec_pr);

    MPI_Finalize();
    return 0;
}