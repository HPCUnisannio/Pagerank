#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define MASTER 0
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
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  AVVIO PAGERANK MPI\n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Nodi: %d - Archi: %d\n", NODES, EDGES);
        printf("║  Processi MPI totali : %d\n", NPROC);
        printf("║  Struttura Dati      : Compressed Sparse Column (CSC)\n");
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }  

    int i, j, k;

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

    double * damp1 = (double*)malloc(NODES*sizeof(double)); //malloc and initialize to 0.85
	double * damp2 = (double*)malloc(NODES*sizeof(double)); //malloc and initialize to 0.15/NODES
	double * diff = (double*)calloc(NODES, sizeof(double));

    for(i=0; i<NODES;i++) {
        damp1[i] = 0.85;
        damp2[i] = 0.15 / NODES;
    }

    double norm, norm_sq;
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

    // Allocate local column pointers for the sparse columns we own
    int *local_colptr = (int*)malloc((pcols[rank] + 1) * sizeof(int));

    // Distribute data
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT, &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Build local column pointers for sparse data
    // Each process needs to know where each column starts in its local rec_val/rec_row
    int local_offset = displs[rank];
    for(i = 0; i < pcols[rank]; i++) {
        int global_col = displs_pr[rank] + i;
        local_colptr[i] = colptr[global_col] - local_offset;
    }
    local_colptr[pcols[rank]] = sendcnts[rank];

    // Phase 2: Power iteration (computation)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute_start, t_compute_end, compute_time;
    if (rank == MASTER) {
        t_setup_end = MPI_Wtime();
        setup_time = t_setup_end - t_setup_start;
        printf("Setup completato in %.6f secondi.\n", setup_time);
        printf("Inizio computazione parallela iterativa...\n");
        t_compute_start = t_setup_end;
    }

    // ========== AGGIUNTA 1: Dichiarazione del contatore ==========
    int iteration_count = 0;  // Contatore delle iterazioni

    do {
        // ========== AGGIUNTA 2: Incremento del contatore ==========
        iteration_count++;  // Incrementa ad ogni iterazione

        memset(sum, 0, NODES * sizeof(double));
        if (rank == MASTER) {
            memset(prnew, 0, NODES * sizeof(double));
        }
        norm = 0.0;

        // Distribute PageRank vector portions
        MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE, rec_pr, pcols[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Sparse matrix-vector multiplication: sum = A * rec_pr
        // Only iterate over non-zero elements we actually have
        for(j = 0; j < pcols[rank]; j++) {
            int col_start = local_colptr[j];
            int col_end = local_colptr[j + 1];
            double pr_val = rec_pr[j];
            
            for(k = col_start; k < col_end; k++) {
                int row = rec_row[k];
                double val = rec_val[k];
                sum[row] += val * pr_val;
            }
        }

        // Reduce partial results to master
        MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

        // Master applies damping, dangling nodes redistribution, and computes convergence
        if (rank == MASTER) {
            double dangling_mass = pagerank_compute_dangling_mass(NODES, prold, readsum);

            for(i=0;i<NODES;i++) {
			    prnew[i]= prnew[i] * damp1[i] + damp2[i] + ((dangling_mass * damp1[i]) / NODES);
			}

			norm_sq = 0.0;
			for(i=0; i<NODES; i++) {// for norm calculation
			    diff[i] = prnew[i] - prold[i];
				norm_sq += diff[i] * diff[i];
				prold[i] = prnew[i];
			}
            
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
        
        /* ----------------------
            PRINT PR SUM CHECK
        ---------------------- */
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  PR SUM CHECK                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Total PR Sum:        %12.10f                              \n", sum_pr);
        printf("║  Expected Sum:        %12.10f                              \n", 1.0);

        double pr_diff = fabs(sum_pr - 1.0);
        printf("║  Difference:          %12.10f                               \n", pr_diff);

        if (pr_diff < 1e-9) {
            printf("║  Status:              ✓ PASSED (within tolerance)         \n");
        } else if (pr_diff < 1e-6) {
            printf("║  Status:              ⚠ WARNING (slightly off)            \n");
        } else {
            printf("║  Status:              ✗ FAILED (significant error)        \n");
        }
        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        /* ----------------------
            TIME PERFORMANCE & METRICS DISPLAY
        ---------------------- */
        // Print execution summary
        char label[100];
        sprintf(label, "MPI BASE (%d processes)", NPROC);
        compute_time = t_compute_end - t_compute_start;
        double total_time = t_compute_end - t_setup_start;
        measure_print_summary(label, setup_time, compute_time, total_time, iteration_count);

        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  LOAD BALANCE                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");

        // Bilanciamento del carico reale calcolato sugli archi (NNZ) assegnati ai processi
        double max_nnz = sendcnts[0];
        double avg_nnz = (double)EDGES / NPROC;
        int r;
        for (r = 1; r < NPROC; r++) {
            if (sendcnts[r] > max_nnz) max_nnz = sendcnts[r];
        }
        double load_balance = measure_load_balance(max_nnz, avg_nnz);
        printf("║  Load Balance (NNZ):  %12.2f%%                             \n", load_balance * 100.0);
        printf("║  Partition Size (NNZ): avg=%.1f, max=%.0f                  \n", avg_nnz, max_nnz);

        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");


        /* ----------------------
            READ SEQUENTIAL TIME & METRICS DISPLAY
        ---------------------- */
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
    free(damp1); free(damp2); free(diff);

    MPI_Finalize();
    return 0;
}