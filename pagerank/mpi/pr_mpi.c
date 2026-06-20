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

    double t_setup_start, t_setup_end, setup_time;
    // Phase 1: Setup - Build CSC matrix from file (all processes read)
    if (rank == MASTER) {
        t_setup_start = MPI_Wtime();
    }
    
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

    // Allocate local receive buffers
    double *rec_val = (double*)malloc(sendcnts[rank] * sizeof(double));
    int    *rec_row = (int*)malloc(sendcnts[rank] * sizeof(int));
    double *rec_pr  = (double*)malloc(pcols[rank] * sizeof(double));

    // Distribute data
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT, &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Phase 2: Power iteration (computation)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute_start, t_compute_end, compute_time;
    if (rank == MASTER) {
        t_setup_end = MPI_Wtime();
        setup_time = t_setup_end - t_setup_start;
        printf("Setup complete\n");
        printf("val, rowind and colptr have been populated\n");
        t_compute_start = t_setup_end;
    }

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

    // Synchronize: ensure ALL processes have exited the loop
    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 3: Validation and output (Master only)
    if (rank == MASTER) {
        t_compute_end = MPI_Wtime();
        
        double sum_pr = pagerank_validate(NODES, prnew);
        printf("=============================================\n");
        printf("VERIFICA MATEMATICA: Somma finale PR = %.10f\n", sum_pr);
        printf("=============================================\n");

        // Print execution summary
        char label[100];
        sprintf(label, "MPI BASE (%d processes)", NPROC);
        compute_time = t_compute_end - t_compute_start;
        double total_time = t_compute_end - t_setup_start;
        measure_print_summary(label, setup_time, compute_time, total_time);

        // Read sequential time for comparison
        double sequential_time = 0.0;
        FILE *time_file = fopen("sequential/sequential_time.txt", "r");
        if (time_file) {
            fscanf(time_file, "%lf", &sequential_time);
            fclose(time_file);
            
            char label1[30];
            sprintf(label1, "Sequenziale");
            char label2[30];
            sprintf(label2, "MPI Base (%d proc)", NPROC);
            
            measure_print_comparison(label1, sequential_time, label2, compute_time);
        } else {
            printf("\nNote: 'sequential_time.txt' not found. Run sequential version first.\n");
        }

        /*
        for(i = 0; i < NODES; i++) {
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